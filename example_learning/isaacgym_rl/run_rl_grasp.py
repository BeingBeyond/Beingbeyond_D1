import os
import json
import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

import gym
from isaacgym import gymapi
from isaacgym import gymutil
import isaacgymenvs
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from isaacgymenvs.utils.torch_jit_utils import *
import tasks

def build_runner(cfg, env):
    train_param = cfg.train.params
    is_testing = cfg.test  # train_param["test"]
    ckpt_path = cfg.checkpoint

    if not is_testing:
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{cfg.task_name}_{time_str}"
        if "run_name" in cfg:
            run_name = cfg.run_name
        log_dir = os.path.join(train_param.log_dir, run_name)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "config.json"), "w") as f:
            json.dump(OmegaConf.to_container(cfg), f, indent=4)
    else:
        log_dir = None

    if train_param.name == "ppo":
        from algo import ppo
        runner = ppo.PPO(
            vec_env=env,
            actor_critic_class=ppo.ActorCritic,
            train_param=train_param,
            log_dir=log_dir,
            apply_reset=False
        )
    else:
        raise ValueError("Unrecognized algorithm!")

    if is_testing and ckpt_path != "":
        print(f"Loading model from {ckpt_path}")
        runner.test(ckpt_path)
    elif ckpt_path != "":
        print(f"\nWarning: load pre-trained policy. Loading model from {ckpt_path}\n")
        runner.load(ckpt_path)

    return runner


@hydra.main(version_base="1.3", config_path="./tasks", config_name="config")
def main(cfg: DictConfig) -> None:
    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(
        cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank
    )

    def create_isaacgym_env(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{cfg.task_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs

    env = create_isaacgym_env()
    env.reset_idx(torch.arange(env.num_envs))
    
    # debug the environment
    if "debug" in cfg: 
        if cfg["debug"] == "check_joint":
            per_joint_duration = 100 * 1
            for t in range(100000):
                act = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float, device=env.device)
                act[:, :env.num_arm_dofs] = env.active_robot_dof_default_pos[:env.num_arm_dofs]
                act[:, env.hand_dof_start_idx:] = env.active_robot_dof_default_pos[env.num_arm_dofs:]
                i_joint = int(t / per_joint_duration) % 12
                i_action_index = i_joint if i_joint < env.num_arm_dofs else i_joint+1
                print(i_joint)
                t_ = t % per_joint_duration
                if t_ < per_joint_duration//2:
                    act[:,i_action_index] = env.robot_dof_lower_limits[env.active_robot_dof_indices[i_joint]]
                else:
                    act[:,i_action_index] = env.robot_dof_upper_limits[env.active_robot_dof_indices[i_joint]]
                act[:, :env.num_arm_dofs] = unscale(
                    act[:, :env.num_arm_dofs],
                    env.robot_dof_lower_limits[env.arm_dof_indices],
                    env.robot_dof_upper_limits[env.arm_dof_indices],
                )
                act[:, env.hand_dof_start_idx:] = unscale(
                    act[:, env.hand_dof_start_idx:],
                    env.robot_dof_lower_limits[env.active_hand_dof_indices],
                    env.robot_dof_upper_limits[env.active_hand_dof_indices],
                )
                obs, _, _, _ = env.step(act)
                print(obs)
        else:
            for t in range(100000):
                action = env.no_op_action
                #action[:, 3] = -1
                #print(action.shape)
                env.step(action)
    
    else:
        runner = build_runner(cfg, env)
        runner.run()

if __name__ == "__main__":
    main()
