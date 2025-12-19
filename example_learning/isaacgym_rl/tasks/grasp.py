'''
The environment to train RL grasping policies
'''
import os, sys, re, pickle
import yaml
import random
import torch
import numpy as np
from torch.nn import functional as F

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask
import math
from copy import deepcopy

class Grasp(VecTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.init_configs(cfg)

        super().__init__(
            self.cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )
        print("num obs: {}, num actions: {}".format(self.num_obs, self.num_acts))

        # viewer camera setup
        if self.viewer != None:
            cam_pos = gymapi.Vec3(1.8, -3.2, 3.0)
            cam_target = gymapi.Vec3(1.8, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _net_cf_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "robot")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        # jacobian entries corresponding to eef
        self.j_eef = jacobian[:, self.arm_eef_index, :, :self.num_arm_dofs]


        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.robot_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, : self.num_robot_dofs
        ]
        self.robot_dof_pos = self.robot_dof_state[..., 0]
        self.robot_dof_vel = self.robot_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
            self.num_envs, -1, 13
        )
        self.num_bodies = self.rigid_body_states.shape[1]
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self.cur_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )

        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.current_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.total_successes = 0
        self.total_resets = 0

        if self.arm_controller == "qpos" and not self.use_relative_control:
            self.no_op_action = unscale(
                self.robot_dof_default_pos.clone().unsqueeze(0).repeat(self.num_envs, 1)[:, self.arm_dof_indices],
                self.robot_dof_lower_limits[self.arm_dof_indices],
                self.robot_dof_upper_limits[self.arm_dof_indices],
            )
            self.no_op_action = torch.cat([
                self.no_op_action,
                torch.zeros((self.num_envs, self.hand_dof_start_idx - self.num_arm_dofs), dtype=torch.float, device=self.device)
            ], dim=-1)
        elif self.arm_controller == "pose":
            self.no_op_action = self.rigid_body_states.view(-1, 13)[self.eef_idx, 0:7].clone()
        else:
            self.no_op_action = torch.zeros(
                (self.num_envs, self.hand_dof_start_idx), dtype=torch.float, device=self.device
            )
        
        self.no_op_action = torch.cat([
            self.no_op_action,
            unscale(
                self.robot_dof_default_pos.clone().unsqueeze(0).repeat(self.num_envs, 1)[:, self.active_hand_dof_indices],
                self.robot_dof_lower_limits[self.active_hand_dof_indices],
                self.robot_dof_upper_limits[self.active_hand_dof_indices],
            )
        ], dim=-1)
        self.delta_action_scale = to_torch(self.cfg["env"]["deltaActionScale"], dtype=torch.float, device=self.device)

    def init_configs(self, cfg):
        self.cfg = cfg
        
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.debug_vis = self.cfg["env"]["enableDebugVis"]

        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.arm_controller = self.cfg["env"]["armController"]
        self.act_max_ang_vel_arm = self.cfg["env"]["actionsMaxAngVelArm"]
        self.act_max_ang_vel_hand = self.cfg["env"]["actionsMaxAngVelHand"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.random_episode_length = self.cfg["env"]["randomEpisodeLength"]
        self.reset_time = -1 #self.cfg["env"].get("resetTime", -1.0)
        assert self.arm_controller in ["qpos", "worlddpose", "eedpose", "pose"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.hand_name = self.cfg["hand_config"]["name"]
        self.hand_specific_cfg = self.cfg["hand_config"] #self.cfg["hand_specific"][self.cfg["hand_name"]]
        self.palm_offset = self.hand_specific_cfg["palm_offset"]
        
        self.num_obs_dict = self.hand_specific_cfg["num_obs_dict"]
        self.obs_type = self.cfg["env"]["observationType"]
        self.cfg["env"]["numObservations"] = \
            sum([(self.num_obs_dict[i] if i in self.obs_type else 0) for i in self.num_obs_dict]) #self.num_obs_dict[self.obs_type]
        #self.cfg["env"]["numObservations"] = self.hand_specific_cfg["numObs"]
        self.cfg["env"]["numStates"] = 0 
        self.cfg["env"]["numActions"] = self.hand_specific_cfg["numActions"]

    def create_sim(self):
        self.dt = self.cfg["sim"]["dt"]
        self.up_axis_idx = 2 if self.cfg["sim"]["up_axis"] == "z" else 1

        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        #self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )

        # if randamizing, apply once immediately on startup before the first sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        # plane_params.static_friction = 1.0
        # plane_params.dynamic_friction = 1.0
        # plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self.asset_root = self.cfg["env"]["asset"]["assetRoot"]       
        self.robot_asset_file = self.hand_specific_cfg["robotAssetFile"]
        robot_asset, robot_dof_props, robot_start_pose = self._prepare_robot_asset(
            self.asset_root, self.robot_asset_file
        )

        self.num_object_shapes = 0
        self.num_object_bodies = 0
        
        # load object
        object_urdf = self.cfg["env"]["asset"]["objectAssetFile"]
        self.object_names = [object_urdf]
        object_asset, _ = self._prepare_object_asset(self.asset_root, object_urdf)
        self.object_fns = [object_urdf]
        
        # load table
        table_asset, table_start_poses = self._prepare_table_asset()

        self.envs = []
        self.eef_idx =  []
        self.robot_indices = []
        self.object_indices = []
        self.table_indices = []
        self.robot_start_states = []
        self.table_start_pos = []

        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # aggregate size
            max_agg_bodies = self.num_robot_bodies + self.num_object_bodies + 1 # robot + object + table
            max_agg_shapes = self.num_robot_shapes + self.num_object_shapes + 1
            if self.aggregate_mode > 0:
               self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # create robot actor
            robot_actor = self.gym.create_actor(
                env_ptr, robot_asset, robot_start_pose, "robot", i, -1 if self.cfg["env"]["enableSelfCollision"] else 1, 1 # seg id=1
            )
            self.robot_start_states.append(
                [robot_start_pose.p.x,robot_start_pose.p.y,robot_start_pose.p.z,
                robot_start_pose.r.x,robot_start_pose.r.y,robot_start_pose.r.z,
                robot_start_pose.r.w,0,0,0,0,0,0,])
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)
            robot_idx = self.gym.get_actor_index(
                env_ptr, robot_actor, gymapi.DOMAIN_SIM
            )
            self.robot_indices.append(robot_idx)

            # add object
            object_handle = self.gym.create_actor(
                env_ptr, object_asset, gymapi.Transform(), "object", i, -1, 2 # seg id=2
            )
            object_idx = self.gym.get_actor_index(
                env_ptr, object_handle, gymapi.DOMAIN_SIM
            )
            self.object_indices.append(object_idx)

            # set object friction
            object_rb_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            for j in range(len(object_rb_props)):
                object_rb_props[j].friction = self.cfg["env"]["objectFriction"]
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_rb_props)

            # add table
            table_handle = self.gym.create_actor(
                env_ptr, table_asset, table_start_poses[i], "table", i, -1, 0
            )
            table_idx = self.gym.get_actor_index(
                env_ptr, table_handle, gymapi.DOMAIN_SIM
            )
            self.table_indices.append(table_idx)
            self.table_start_pos.append([table_start_poses[i].p.x, table_start_poses[i].p.y, table_start_poses[i].p.z])

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            eef_idx = self.gym.find_actor_rigid_body_index(
                env_ptr, robot_actor, self.hand_specific_cfg["eef_link"], gymapi.DOMAIN_SIM
            )
            self.eef_idx.append(eef_idx)

        self.robot_start_states = to_torch(
            self.robot_start_states, device=self.device
        ).view(num_envs, 13)
        self.object_init_states = to_torch(
            self.object_init_states, device=self.device
        ).view(num_envs, 13)
        self.fingertip_handles = to_torch(
            self.fingertip_handles, dtype=torch.long, device=self.device
        )
        self.palm_handle = to_torch(
            self.palm_handle, dtype=torch.long, device=self.device
        )
        self.robot_indices = to_torch(
            self.robot_indices, dtype=torch.long, device=self.device
        )
        self.object_indices = to_torch(
            self.object_indices, dtype=torch.long, device=self.device
        )
        self.table_indices = to_torch(
            self.table_indices, dtype=torch.long, device=self.device
        )
        self.eef_idx = to_torch(self.eef_idx, dtype=torch.long, device=self.device)
        self.reset_position_range = to_torch(self.cfg["env"]["resetPositionRange"], dtype=torch.float, device=self.device)
        self.reset_random_rot = self.cfg["env"]["resetRandomRot"]
        self.table_height_range = to_torch(self.cfg["env"]["tableHeightRange"], dtype=torch.float, device=self.device)
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.table_start_pos = to_torch(
            self.table_start_pos, dtype=torch.float, device=self.device
        )

    def _prepare_robot_asset(self, asset_root, asset_file):
        # load robot asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.collapse_fixed_joints = False
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        if self.cfg["env"]["useRobotVhacd"]:
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 300000
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        # drive_mode: 0: none, 1: position, 2: velocity, 3: force
        asset_options.default_dof_drive_mode = 0

        print("Loading robot asset: ", asset_root, asset_file)
        #asset_file = 'robot/inspire_tac/fr3_inspire_tac_L_right_safety.urdf'
        #exit(0)
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # get asset info
        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)
        #self.num_robot_actuators = self.num_robot_dofs
        print("self.num_robot_bodies: ", self.num_robot_bodies)
        print("self.num_robot_shapes: ", self.num_robot_shapes)
        print("self.num_robot_dofs: ", self.num_robot_dofs)
        #print("self.num_robot_actuators: ", self.num_robot_actuators)

        self.palm = self.hand_specific_cfg["palm_link"] #"palm_lower"
        self.fingertips = self.hand_specific_cfg["fingertips_link"]
        self.num_fingers = len(self.fingertips)
        self.num_arm_dofs = self.hand_specific_cfg["num_arm_dofs"]
        self.robot_dof_names = []
        for i in range(self.num_robot_dofs):
            joint_name = self.gym.get_asset_dof_name(robot_asset, i)
            self.robot_dof_names.append(joint_name)
        
        self.palm_handle = self.gym.find_asset_rigid_body_index(robot_asset, self.palm)
        self.fingertip_handles = [
            self.gym.find_asset_rigid_body_index(robot_asset, fingertip)
            for fingertip in self.fingertips
        ]
        if -1 in self.fingertip_handles or self.palm_handle==-1:
            raise Exception("Fingertip names or palm name not found!")
        self.arm_dof_names = self.hand_specific_cfg["arm_dof_names"]
        self.arm_dof_indices = [
            self.gym.find_asset_dof_index(robot_asset, name)
            for name in self.arm_dof_names
        ]
        self.hand_dof_names = []
        for name in self.robot_dof_names:
            if name not in self.arm_dof_names:
                self.hand_dof_names.append(name)
        self.hand_dof_indices = [
            self.gym.find_asset_dof_index(robot_asset, name)
            for name in self.hand_dof_names
        ]
        self.robot_dof_indices = self.arm_dof_indices + self.hand_dof_indices
        self.robot_dof_indices = to_torch(
            self.robot_dof_indices, dtype=torch.long, device=self.device
        )
        self.hand_dof_start_idx = 7 #len(self.arm_dof_indices)

        # process tendon joints
        if "passive_joints" in self.hand_specific_cfg:
            self.have_passive_joints = True
            self.passive_hand_dof_indices, self.mimic_parent_dof_indices, self.mimic_multipliers = [],[],[]
            for k, v in self.hand_specific_cfg["passive_joints"].items():
                self.passive_hand_dof_indices.append(self.gym.find_asset_dof_index(robot_asset, k))
                self.mimic_parent_dof_indices.append(self.gym.find_asset_dof_index(robot_asset, v["mimic"]))
                self.mimic_multipliers.append(v["multiplier"])
            #print("Passive joints:", self.hand_specific_cfg["passive_joints"])
            #print(self.passive_hand_dof_indices, self.mimic_parent_dof_indices, self.mimic_multipliers)
            self.active_hand_dof_indices = []
            for i in self.hand_dof_indices:
                if i not in self.passive_hand_dof_indices:
                    self.active_hand_dof_indices.append(i)
            #print(self.active_hand_dof_indices)
            self.mimic_multipliers = to_torch(self.mimic_multipliers, device=self.device)
        else:
            self.have_passive_joints = False
            self.active_hand_dof_indices = self.hand_dof_indices
        self.active_robot_dof_indices = self.arm_dof_indices + self.active_hand_dof_indices
        self.active_robot_dof_indices = to_torch(
            self.active_robot_dof_indices, dtype=torch.long, device=self.device
        )
        self.active_robot_dof_names = [self.robot_dof_names[i] for i in self.active_robot_dof_indices]
        print("Active dof names:", self.active_robot_dof_names)
        self.active_hand_dof_names = [self.robot_dof_names[i] for i in self.active_hand_dof_indices]
        print("Active hand dof names:", self.active_hand_dof_names)
        print("Hand dof names:", self.hand_dof_names)

        # count dofs
        assert self.arm_dof_indices == [i for i in range(self.num_arm_dofs)]
        print("arm dof indices, active hand dof indices, hand dof start idx:", \
              self.arm_dof_indices, self.active_hand_dof_indices, self.hand_dof_start_idx)
        assert self.num_arm_dofs == len(self.arm_dof_indices)
        self.num_hand_dofs = len(self.hand_dof_indices)
        self.num_active_hand_dofs = len(self.active_hand_dof_indices)
        self.num_passive_hand_dofs = len(self.passive_hand_dof_indices) if self.have_passive_joints else 0
        #self.num_active_robot_dofs = self.num_arm_dofs + self.num_active_hand_dofs
        assert self.num_arm_dofs+self.num_hand_dofs==self.num_robot_dofs

        # get eef index
        robot_link_dict = self.gym.get_asset_rigid_body_dict(robot_asset)
        self.arm_eef_index = robot_link_dict[self.hand_specific_cfg["eef_link"]]

        # dof properties
        self.default_dof_pos = np.array(self.hand_specific_cfg["default_dof_pos"], dtype=np.float32)
        print("Default DoF positions: ", self.default_dof_pos)
        assert self.num_robot_dofs == len(self.default_dof_pos)
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.robot_dof_lower_limits = []
        self.robot_dof_upper_limits = []
        self.robot_dof_default_pos = []
        self.robot_dof_default_vel = []

        for i in range(self.num_robot_dofs):
            self.robot_dof_lower_limits.append(robot_dof_props["lower"][i])
            self.robot_dof_upper_limits.append(robot_dof_props["upper"][i])
            self.robot_dof_default_pos.append(self.default_dof_pos[i])
            self.robot_dof_default_vel.append(0.0)
        
            # large kp, kd to simulate position control
            if i in self.arm_dof_indices:
                robot_dof_props["driveMode"][i] = 1
                robot_dof_props["stiffness"][i] = 1000
                robot_dof_props["damping"][i] = 20
                robot_dof_props["friction"][i] = 0.01
                robot_dof_props["armature"][i] = 0.001
            elif i in self.hand_dof_indices:
                robot_dof_props["driveMode"][i] = 1
                robot_dof_props["stiffness"][i] = 3
                robot_dof_props["damping"][i] = 0.5
                robot_dof_props["friction"][i] = 0.01
                robot_dof_props["armature"][i] = 0.001
            print('DoF {} effort {:.2} stiffness {:.2} damping {:.2} friction {:.2} armature {:.2} limit {:.2}~{:.2}'.format(
                self.robot_dof_names[(self.arm_dof_indices + self.hand_dof_indices).index(i)], 
                robot_dof_props['effort'][i], robot_dof_props['stiffness'][i],
                robot_dof_props['damping'][i], robot_dof_props['friction'][i],
                robot_dof_props['armature'][i], robot_dof_props['lower'][i], 
                robot_dof_props['upper'][i]))


        self.robot_dof_lower_limits = to_torch(self.robot_dof_lower_limits, device=self.device)
        self.robot_dof_upper_limits = to_torch(self.robot_dof_upper_limits, device=self.device)
        self.robot_dof_default_pos = to_torch(self.robot_dof_default_pos, device=self.device)
        self.active_robot_dof_default_pos = self.robot_dof_default_pos[:self.num_arm_dofs + self.num_active_hand_dofs]
        self.robot_dof_default_vel = to_torch(self.robot_dof_default_vel, device=self.device)
        print(f"Arm DoF limits: {[(i.item(),j.item()) for (i,j) in zip(self.robot_dof_lower_limits[self.arm_dof_indices], self.robot_dof_upper_limits[self.arm_dof_indices])]}")
        print(f"Hand DoF limits: {[(i.item(),j.item()) for (i,j) in zip(self.robot_dof_lower_limits[self.hand_dof_indices], self.robot_dof_upper_limits[self.hand_dof_indices])]}")
        print(f"Active Hand Dof Lower: {self.robot_dof_lower_limits[self.active_hand_dof_indices]}")
        print(f"Active Hand Dof Upper: {self.robot_dof_upper_limits[self.active_hand_dof_indices]}")

        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(0, 0, 0)
        robot_start_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)
        #print(robot_start_pose.p, robot_start_pose.r)
        return robot_asset, robot_dof_props, robot_start_pose


    def _prepare_object_asset(self, asset_root, asset_file):
        # load object asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.collapse_fixed_joints = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        if self.cfg["env"]["useObjectVhacd"]:
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 300000

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True

        # drive_mode: 0: none, 1: position, 2: velocity, 3: force
        asset_options.default_dof_drive_mode = 0

        object_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

        # get object asset info
        self.num_object_bodies = max(self.num_object_bodies, self.gym.get_asset_rigid_body_count(object_asset))
        self.num_object_shapes = max(self.num_object_shapes, self.gym.get_asset_rigid_shape_count(object_asset))
        self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)

        object_dof_props = self.gym.get_asset_dof_properties(object_asset)
        self.object_dof_lower_limits = []
        self.object_dof_upper_limits = []
        for i in range(self.num_object_dofs):
            self.object_dof_lower_limits.append(object_dof_props["lower"][i])
            self.object_dof_upper_limits.append(object_dof_props["upper"][i])

        self.object_dof_lower_limits = to_torch(
            self.object_dof_lower_limits, device=self.device
        )
        self.object_dof_upper_limits = to_torch(
            self.object_dof_upper_limits, device=self.device
        )
        self.object_init_states = to_torch([0.,0.,0.1, 0.,0.,0.,1., 0.,0.,0.,0.,0.,0.], dtype=torch.float, device=self.device).repeat(self.num_envs, 1)
        
        return object_asset, object_dof_props
    
    def _prepare_table_asset(self):
        self.table_thickness = 0.3
        self.table_heights = to_torch(self.cfg["env"]["tableHeightRange"][0], dtype=torch.float, device=self.device).repeat(self.num_envs)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        
        table_dims = gymapi.Vec3(0.6, 0.6, self.table_thickness)
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(0.3, 0, self.table_heights[0] - self.table_thickness/2)
        table_start_poses = [table_start_pose] * self.num_envs
        table_asset = self.gym.create_box(
            self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options
        )
        return table_asset, table_start_poses


    def reset_idx(self, env_ids, object_init_pose=None, **kwargs):
        ## randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        ## reset object
        # apply random rotation
        rand_rot_axis = np.random.randn(len(env_ids), 3)
        if self.reset_random_rot == "z":
            rand_rot_axis[:] = np.array([0, 0, 1])
        rand_rot_axis = to_torch(rand_rot_axis / np.linalg.norm(rand_rot_axis, axis=1, keepdims=True), device=self.device)
        rand_angle = torch_rand_float(-np.pi, np.pi, (len(env_ids), 1), device=self.device)
        if self.reset_random_rot == "fixed":
            rand_angle[:] = 0.0
        self.root_state_tensor[self.object_indices[env_ids], 3:7] = (
            quat_from_angle_axis(rand_angle[:,-1], rand_rot_axis)
        )
        # sample random xyz
        samples = self.reset_position_range[:, 0] + (self.reset_position_range[:, 1] - self.reset_position_range[:, 0]) * torch.rand(len(env_ids), 3).to(self.device)
        samples[:, 2] += self.table_heights[env_ids] # add table height
        self.root_state_tensor[self.object_indices[env_ids], 0:3] = samples
        self.root_state_tensor[self.object_indices[env_ids], 7:] = 0
        # if use predefined object pose
        if object_init_pose is not None:
            self.root_state_tensor[self.object_indices[env_ids], 0:7] = to_torch(object_init_pose, device=self.device)

        table_object_indices = torch.cat(
            [self.table_indices[env_ids], self.object_indices[env_ids]], dim=0
        ).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(table_object_indices),
            len(table_object_indices),
        )

        ## reset robot
        delta_max = self.robot_dof_upper_limits - self.robot_dof_default_pos
        delta_min = self.robot_dof_lower_limits - self.robot_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (
            torch_rand_float(-1.0,1.0,(len(env_ids),self.num_robot_dofs),device=self.device) + 1.0
        )
        pos = self.robot_dof_default_pos + self.reset_dof_pos_noise * rand_delta
        self.robot_dof_pos[env_ids, :] = pos
        self.robot_dof_vel[env_ids, :] = self.robot_dof_default_vel
        self.prev_targets[env_ids, : self.num_robot_dofs] = pos.clone()
        self.cur_targets[env_ids, : self.num_robot_dofs] = pos.clone()

        robot_indices = self.robot_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(robot_indices),
            len(env_ids),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.robot_dof_state),
            gymtorch.unwrap_tensor(robot_indices),
            len(env_ids),
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float, device=self.device)
        
        self.compute_observations()

    def pre_physics_step(self, actions):
        # reset when done
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            #print(env_ids)
            self.reset_idx(env_ids)
          
        #print(actions)
        self.actions = actions.clone().to(self.device)
        
        if self.use_relative_control:
            # last qpos + delta action
            self.cur_targets[:, self.active_hand_dof_indices] = \
                self.prev_targets[:, self.active_hand_dof_indices] + \
                    self.actions[:, self.hand_dof_start_idx:] * self.delta_action_scale[self.hand_dof_start_idx:]
        else:
            # [-1,1] action -> target qpos
            self.cur_targets[:, self.active_hand_dof_indices] = scale(
                self.actions[:, self.hand_dof_start_idx:],
                self.robot_dof_lower_limits[self.active_hand_dof_indices],
                self.robot_dof_upper_limits[self.active_hand_dof_indices],
            )
            # moving average
            self.cur_targets[:, self.active_hand_dof_indices] = (
                self.act_moving_average * self.cur_targets[:, self.active_hand_dof_indices]
                + (1.0 - self.act_moving_average) * self.prev_targets[:, self.active_hand_dof_indices]
            )
        # clip to satisfy max step size
        # self.cur_targets[:, self.active_hand_dof_indices] = tensor_clamp(
        #     self.cur_targets[:, self.active_hand_dof_indices],
        #     self.prev_targets[:, self.active_hand_dof_indices]
        #     - self.act_max_ang_vel_hand * self.dt,
        #     self.prev_targets[:, self.active_hand_dof_indices]
        #     + self.act_max_ang_vel_hand * self.dt,
        # )
        # set passive joints
        if self.have_passive_joints:
                #print(self.active_hand_dof_indices, self.passive_hand_dof_indices, self.mimic_parent_dof_indices, self.mimic_multipliers)
                #print(self.robot_dof_lower_limits[self.robot_dof_indices], self.robot_dof_upper_limits[self.robot_dof_indices])
                self.cur_targets[:, self.passive_hand_dof_indices] = \
                    self.cur_targets[:, self.mimic_parent_dof_indices] * self.mimic_multipliers
        # clip to joint limits
        self.cur_targets[:, self.hand_dof_indices] = tensor_clamp(
            self.cur_targets[:, self.hand_dof_indices],
            self.robot_dof_lower_limits[self.hand_dof_indices],
            self.robot_dof_upper_limits[self.hand_dof_indices],
        )

        if self.arm_controller == "qpos":
            if self.use_relative_control:
                print("Warning: Currently, relative control for arm is implemented as direct qpos copy!!!")
                self.cur_targets[:, self.arm_dof_indices] = scale(
                    self.actions[:, :self.hand_dof_start_idx],
                    self.robot_dof_lower_limits[self.arm_dof_indices],
                    self.robot_dof_upper_limits[self.arm_dof_indices],
                )
                ## last qpos + delta action
                #self.cur_targets[:, self.arm_dof_indices] = self.prev_targets[:, self.arm_dof_indices] + self.actions[:, :self.hand_dof_start_idx]
            else:
                # [-1,1] action -> target qpos
                self.cur_targets[:, self.arm_dof_indices] = scale(
                    self.actions[:, :self.num_arm_dofs],
                    self.robot_dof_lower_limits[self.arm_dof_indices],
                    self.robot_dof_upper_limits[self.arm_dof_indices],
                )
                # moving average
                self.cur_targets[:, self.arm_dof_indices] = (
                    self.act_moving_average * self.cur_targets[:, self.arm_dof_indices]
                    + (1.0 - self.act_moving_average) * self.prev_targets[:, self.arm_dof_indices]
                )
            
            # clip to satisfy max step size
            # self.cur_targets[:, self.arm_dof_indices] = tensor_clamp(
            #     self.cur_targets[:, self.arm_dof_indices],
            #     self.prev_targets[:, self.arm_dof_indices]
            #     - self.act_max_ang_vel_arm * self.dt,
            #     self.prev_targets[:, self.arm_dof_indices]
            #     + self.act_max_ang_vel_arm * self.dt,
            # )
            # clip to joint limits
            self.cur_targets[:, self.arm_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.arm_dof_indices],
                self.robot_dof_lower_limits[self.arm_dof_indices],
                self.robot_dof_upper_limits[self.arm_dof_indices],
            )
        elif "pose" in self.arm_controller:
            if self.arm_controller == "pose":
                # absolute pose control
                delta_arm_action = self.compute_arm_ik(self.actions[:, :7], is_delta_pose=False)
            else:
                # delta pose control
                delta_arm_action = self.compute_arm_ik(self.actions[:, :6] * self.delta_action_scale[:6], is_delta_pose=True, is_delta_pose_in_world=("world" in self.arm_controller))
            self.cur_targets[:, self.arm_dof_indices] = self.robot_dof_pos[:, self.arm_dof_indices] + delta_arm_action
            # clip to satisfy max step size
            self.cur_targets[:, self.arm_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.arm_dof_indices],
                self.prev_targets[:, self.arm_dof_indices]
                - self.act_max_ang_vel_arm * self.dt,
                self.prev_targets[:, self.arm_dof_indices]
                + self.act_max_ang_vel_arm * self.dt,
            )
            # clip to joint limits
            self.cur_targets[:, self.arm_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.arm_dof_indices],
                self.robot_dof_lower_limits[self.arm_dof_indices],
                self.robot_dof_upper_limits[self.arm_dof_indices],
            )
        
        self.prev_targets[:, self.robot_dof_indices] = self.cur_targets[
            :, self.robot_dof_indices
        ]
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.cur_targets)
        )


    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1
        self.compute_observations()
        self.compute_reward()

        if self.viewer and self.debug_vis:
            # draw axes to debug
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            object_state = self.root_state_tensor[self.object_indices, :]
            for i in range(self.num_envs):
                self._add_debug_lines(
                    self.envs[i], object_state[i, :3], object_state[i, 3:7]
                )
                self._add_debug_lines(
                    self.envs[i], self.palm_center_pos[i], self.palm_rot[i]
                )
                for j in range(self.num_fingers):
                    self._add_debug_lines(
                        self.envs[i],
                        self.fingertip_pos[i][j],
                        self.fingertip_rot[i][j],
                    )


    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.palm_state = self.rigid_body_states[:, self.palm_handle][..., :13]
        self.palm_pos = self.palm_state[..., :3]
        self.palm_rot = self.palm_state[..., 3:7]
        self.palm_center_pos = self.palm_pos + quat_apply(
            self.palm_rot, to_torch(self.palm_offset).repeat(self.num_envs, 1)
        )
        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][
            ..., :13
        ]
        self.fingertip_pos = self.fingertip_state[..., :3]
        self.fingertip_rot = self.fingertip_state[..., 3:7]

        self.compute_required_observations(self.obs_buf, self.obs_type, self.num_observations)
        
    # compute obs with required contents
    def compute_required_observations(self, obs_buf, obs_type, num_obs):
        obs_end = 0

        if "armdof" in obs_type:
            obs_buf[:, obs_end: obs_end+self.num_arm_dofs] = unscale(
                self.robot_dof_pos[:, self.arm_dof_indices],
                self.robot_dof_lower_limits[self.arm_dof_indices],
                self.robot_dof_upper_limits[self.arm_dof_indices],
            )
            obs_end += self.num_arm_dofs

        if "handdof" in obs_type:
            obs_buf[:, obs_end: obs_end+self.num_active_hand_dofs] = unscale(
                self.robot_dof_pos[:, self.active_hand_dof_indices],
                self.robot_dof_lower_limits[self.active_hand_dof_indices],
                self.robot_dof_upper_limits[self.active_hand_dof_indices],
            )
            obs_end += self.num_active_hand_dofs

        if "eefpose" in obs_type:
            obs_buf[:, obs_end: obs_end+7] = self.rigid_body_states.view(-1, 13)[self.eef_idx, 0:7]
            obs_end += 7

        if "ftpos" in obs_type: # fingertip positions, N*3
            num_ft_states = self.num_fingers * 3
            obs_buf[:, obs_end: obs_end+num_ft_states] = (
                self.fingertip_pos.reshape(self.num_envs, num_ft_states)
            )
            obs_end += num_ft_states
                
        if "palmpose" in obs_type: # palm pose, N*7
            obs_buf[:, obs_end: obs_end+3] = self.palm_pos
            obs_buf[:, obs_end+3: obs_end+7] = self.palm_rot
            obs_end += 7
        
        if "handposerror" in obs_type: # joint position control error of hand
            obs_buf[:, obs_end: obs_end+self.num_active_hand_dofs] = \
                (self.cur_targets[:, self.active_hand_dof_indices] - self.robot_dof_pos[:, self.active_hand_dof_indices])
            obs_end += self.num_active_hand_dofs

        if "lastact" in obs_type: # last action
            obs_buf[:, obs_end : obs_end+self.num_actions] = self.actions
            obs_end += self.num_actions

        if "objpose" in obs_type: # object pose: pos, rot (7)
            obs_buf[:, obs_end: obs_end+7] = self.object_pose
            obs_end += 7
        
        assert obs_end == num_obs

    def _add_debug_lines(self, env, pos, rot, line_len=0.2):
        posx = (
            (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * line_len))
            .cpu()
            .numpy()
        )
        posy = (
            (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * line_len))
            .cpu()
            .numpy()
        )
        posz = (
            (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * line_len))
            .cpu()
            .numpy()
        )

        p0 = pos.cpu().numpy()
        self.gym.add_lines(
            self.viewer,
            env,
            1,
            [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]],
            [0.85, 0.1, 0.1],
        )
        self.gym.add_lines(
            self.viewer,
            env,
            1,
            [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]],
            [0.1, 0.85, 0.1],
        )
        self.gym.add_lines(
            self.viewer,
            env,
            1,
            [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]],
            [0.1, 0.1, 0.85],
        )

    ### convert (delta) target ee pose to delta joint angles of the arm
    def compute_arm_ik(self, action, is_delta_pose=True, is_delta_pose_in_world=True, reference_state=None):
        '''
        action: either 6-dim pos+angle-axis for delta pose, or 7-dim pos+quat for absolute target pose
        is_delta_pose: delta pose or absolute target pose?
        is_delta_pose_in_world: delta pose is in the world frame or in the current end-effector frame?
        reference_state: current end-effector state by default
        '''
        if reference_state is None:
            reference_state = self.rigid_body_states.view(-1, 13)[self.eef_idx, 0:7]

        # delta action: 3 dim delta position + 3 dim delta angle-axis
        if is_delta_pose:
            delta_action = action
            if is_delta_pose_in_world:
                # delta pose defined in the world frame
                pos_err = delta_action[:, 0:3]
                dtheta = torch.norm(delta_action[:, 3:6], dim=-1, keepdim=True)
                axis = delta_action[:, 3:6] / (dtheta + 1e-4)
                delta_quat = quat_from_angle_axis(dtheta.squeeze().view(-1), axis)
                orn_err = orientation_error(
                    quat_mul(delta_quat, reference_state[:, 3:7]),
                    reference_state[:, 3:7],
                )
            else:
                # delta pose defined in the end-effector frame
                pos_err = quat_apply(reference_state[:, 3:7], delta_action[:, 0:3])
                dtheta = torch.norm(delta_action[:, 3:6], dim=-1, keepdim=True)
                axis = delta_action[:, 3:6] / (dtheta + 1e-4)
                delta_quat = quat_from_angle_axis(dtheta.squeeze().view(-1), axis)
                orn_err = orientation_error(
                    quat_mul(reference_state[:, 3:7], delta_quat),
                    reference_state[:, 3:7],
                )
        # absolute target action: 3 dim position + 4 dim quat
        else:
            pos_err = action[:, 0:3] - reference_state[:, 0:3]
            orn_err = orientation_error(quat_unit(action[:, 3:7]), reference_state[:, 3:7])
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        u = self._control_ik(dpose) # the input dpose of _control_ik is always in the world (base) frame
        return u    
    
    def _control_ik(self, dpose):
        damping = 0.1
        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        #print(j_eef_T.shape)
        lmbda = torch.eye(6, device=self.device) * (damping**2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(
            self.num_envs, self.num_arm_dofs
        )
        return u

    def compute_reward(self):
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.progress_buf[:],
            self.successes[:],
            self.current_successes[:],
            self.consecutive_successes[:],
            reward_info,
        ) = compute_task_rewards(
            self.reset_buf,
            self.progress_buf,
            self.successes,
            self.current_successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos,
            0.3, # goal height
            self.palm_center_pos,
            self.fingertip_pos,
            torch.as_tensor(self.num_fingers).to(self.device),
            self.object_init_states,
            0.1, # success tolerance
            0.1, # av factor
        )

        self.extras.update(reward_info)
        self.extras["successes"] = self.successes
        self.extras["current_successes"] = self.current_successes
        self.extras["consecutive_successes"] = self.consecutive_successes


@torch.jit.script
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def compute_task_rewards(
    reset_buf,
    progress_buf,
    successes,
    current_successes,
    consecutive_successes,
    max_episode_length: float,
    object_pos,
    goal_height: float,
    palm_pos,
    fingertip_pos,
    num_fingers,
    object_init_states,
    success_tolerance: float,
    av_factor: float,
):
    info = {}

    goal_object_dist = torch.abs(goal_height - object_pos[:, 2])
    palm_object_dist = torch.norm(object_pos - palm_pos, dim=-1)
    palm_object_dist = torch.where(palm_object_dist >= 0.5, 0.5, palm_object_dist)
    #horizontal_offset = torch.norm(
    #    object_pos[:, 0:2] - object_init_states[:, 0:2], dim=-1
    #)

    fingertips_object_dist = torch.zeros_like(goal_object_dist)
    for i in range(fingertip_pos.shape[-2]):
        fingertips_object_dist += torch.norm(
            fingertip_pos[:, i, :] - object_pos, dim=-1
        )
    fingertips_object_dist = torch.where(
        fingertips_object_dist >= 3.0, 3.0, fingertips_object_dist
    )
    fingertips_object_dist = fingertips_object_dist / num_fingers

    thumb_index_middle_pos = (fingertip_pos[:, 0] + fingertip_pos[:, 1]) / 2.0
    middle_point_object_dist = torch.norm(
        thumb_index_middle_pos - object_pos, dim=-1
    )
    middle_point_object_dist = torch.where(
        middle_point_object_dist >= 0.5, 0.5, middle_point_object_dist
    )

    object_to_two_fingers_cosine = torch.nn.functional.cosine_similarity(
        fingertip_pos[:, 0] - object_pos,
        fingertip_pos[:, 1] - object_pos,
        dim=-1,
    )
    #print(object_to_two_fingers_cosine)

    flag = (fingertips_object_dist <= 0.12) + (palm_object_dist <= 0.15) + (middle_point_object_dist <= 0.08)
    # object lift reward
    object_lift_reward = torch.zeros_like(goal_object_dist)
    object_lift_reward = torch.where(
        flag >= 1, 1 * (0.9 - 3 * goal_object_dist), object_lift_reward
    )
    # lift to goal bonus
    bonus = torch.zeros_like(goal_object_dist)
    bonus = torch.where(
        flag >= 1,
        torch.where(
            goal_object_dist <= success_tolerance, 1.0 / (1 + goal_object_dist), bonus
        ),
        bonus,
    )

    resets = reset_buf.clone()
    resets = torch.where(
        progress_buf >= max_episode_length, torch.ones_like(resets), resets
    )
    object_fall = (object_pos[:, 2] <= 0).to(torch.float32)
    resets = torch.where(object_fall > 0, torch.ones_like(resets), resets) # object fall
    successes = torch.where(
        object_pos[:, 2] - goal_height >= -success_tolerance, #goal_object_dist <= success_tolerance,
        torch.where(
            flag >= 1, torch.ones_like(successes), successes
        ),
        torch.zeros_like(successes),
    )

    task_successes = torch.zeros_like(goal_object_dist)
    task_successes = torch.where(
        torch.logical_and(resets > 0, successes > 0),
        torch.ones_like(task_successes),
        torch.zeros_like(task_successes),
    )

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    #print(resets.dtype, successes.dtype, current_successes.dtype, consecutive_successes.dtype)
    current_successes = torch.where(resets>0, successes, current_successes)
    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets
        + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    # TODO reward shaping
    reward = (
        - 2.0 * fingertips_object_dist
        - palm_object_dist
        - 2.0 * middle_point_object_dist
        #- 0.3 * object_to_two_fingers_cosine
        + 3 * object_lift_reward
        + bonus
        #- 0.3 * horizontal_offset
        - 100 * object_fall
    )

    info["fingertips_object_dist"] = fingertips_object_dist
    info["palm_object_dist"] = palm_object_dist
    info["middle_point_object_dist"] = middle_point_object_dist
    info["object_to_two_fingers_cosine"] = object_to_two_fingers_cosine
    info["object_lift_reward"] = object_lift_reward
    info["bonus"] = bonus
    #info["horizontal_offset"] = horizontal_offset
    info["object_fall"] = object_fall
    info["reward"] = reward
    info["hand_approach_flag"] = flag

    return (
        reward,
        resets,
        progress_buf,
        successes,
        current_successes,
        cons_successes,
        info,
    )