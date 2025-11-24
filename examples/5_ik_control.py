"""
Example 5: IK to a target EE pose and execute on real hardware (iterative).

This script:
  - Uses D1Kinematics to query the current end-effector pose
  - Builds a target pose in the base frame
  - Iteratively solves IK with arm-only (head fixed at current config)
  - Commands each intermediate solution to the physical D1 head+arm

Use this example to:
  - Verify that the kinematics model matches the hardware
  - Check that IK solutions are feasible and stable
  - Inspect how small Cartesian offsets map to joint motion.
"""

import time
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

from beingbeyond_d1_sdk import HeadArmRobot
from beingbeyond_d1_sdk.pin_kinematics import D1Kinematics, D1KinematicsConfig


def deg_list_to_rad(xs):
    return [math.radians(v) for v in xs]


def rad_list_to_deg(xs):
    return [math.degrees(v) for v in xs]


def main():
    print("\033[91mWARNING: Always keep the physical emergency stop button within reach.\033[0m")
    print("\033[91m         Press it immediately if the robot motion looks unsafe.\033[0m\n")

    from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path
    urdf = get_default_urdf_path()
    dev = "/dev/ttyUSB0"

    # 1) Construct kinematics
    kin_cfg = D1KinematicsConfig(urdf_path=urdf)
    kin = D1Kinematics(kin_cfg)

    # 2) Connect to real robot
    with HeadArmRobot(urdf_path=urdf, dev=dev, baudrate=1_000_000) as robot:
        joint_names = robot.joint_names
        n_joints = len(joint_names)
        print("[Info] Joint order:", joint_names)

        # 2.1) Set a initial posture
        q_init_deg = [
            0.0, 0.0,          # head_yaw, head_pitch
            0.0, -60.0, 60.0,  # shoulder / elbow
            0.0, 0.0, 0.0,     # wrist joints
        ]
        q_init_rad = deg_list_to_rad(q_init_deg)
        robot.set_positions(q_init_rad)
        robot.wait_until_reached(q_init_rad, active_joint_indices=range(n_joints))
        time.sleep(0.5)

        # 3) Read current joint state and compute current EE pose
        q_current = np.asarray(robot.get_positions(), dtype=float)  # shape (8,)
        q_head, q_arm = kin.split_q(q_current)

        T_base_ee = kin.ee_in_base(q_head, q_arm)
        print("[Info] Current EE T in base frame:\n", T_base_ee)

        # 4) Define a target quatpose in base frame
        p_target = np.array([0.2, 0.1, 0.2], dtype=float)
        rpy_target = np.array([0.0, np.pi / 2, 0.0], dtype=float)
        q_target_xyzw = R.from_euler("xyz", rpy_target).as_quat()  # SciPy: [x, y, z, w]

        target_quatpose = np.zeros(7, dtype=float)
        target_quatpose[0:3] = p_target
        target_quatpose[3:7] = q_target_xyzw  # D1Kinematics 按 xyzw 解释

        # 5) IK
        max_outer_iters = 30
        pos_tol = 0.02                     # meters
        rot_tol = np.deg2rad(10.0)         # radians

        q_head_iter = q_head.copy()
        q_arm_iter = q_arm.copy()

        print("\n[IK] Starting iterative IK ...")
        for outer in range(max_outer_iters):
            # 5.1) Forward kinematics with current q
            T_cur = kin.ee_in_base(q_head_iter, q_arm_iter)
            p_cur = T_cur[:3, 3]
            R_cur = T_cur[:3, :3]
            q_cur_xyzw = R.from_matrix(R_cur).as_quat()

            # 5.2) Compute pose error
            dp = p_target - p_cur
            pos_err = np.linalg.norm(dp)

            dR = R.from_quat(q_target_xyzw) * R.from_quat(q_cur_xyzw).inv()
            rot_err = np.linalg.norm(dR.as_rotvec())

            print(
                f"[IK][{outer:02d}] pos_err={pos_err:.3f} m, "
                f"rot_err={np.rad2deg(rot_err):.1f} deg"
            )

            # 5.3) Check convergence
            if pos_err < pos_tol and rot_err < rot_tol:
                print("[IK] Converged to target pose.")
                break

            # 5.4) One IK update step (arm-only, head fixed)
            q_head_sol, q_arm_sol, cost, inner_iters = kin.ik_ee_quatpose_with_arm_only(
                target_ee_quatpose=target_quatpose,
                q_head=q_head_iter,
                q_arm_init=q_arm_iter,
            )
            print(
                f"[IK][{outer:02d}] step cost={cost:.6f}, "
                f"inner_iters={inner_iters}"
            )
            print("[IK] head (rad):", q_head_sol)
            print("[IK] arm  (deg):", rad_list_to_deg(q_arm_sol.tolist()))

            # 5.5) Update current solution
            q_head_iter = q_head_sol
            q_arm_iter = q_arm_sol

            # 5.6) Send this step to the robot
            q_cmd = np.concatenate([q_head_iter, q_arm_iter])
            robot.set_positions(q_cmd)

        else:
            print("[IK] WARNING: Max outer iterations reached without convergence.")

        # 6) Final FK check
        T_final = kin.ee_in_base(q_head_iter, q_arm_iter)
        p_final = T_final[:3, 3]
        R_final = T_final[:3, :3]
        q_final_xyzw = R.from_matrix(R_final).as_quat()

        print("\n[Check] p_target:", p_target)
        print("[Check] p_final :", p_final)
        print("[Check] q_target (xyzw):", q_target_xyzw)
        print("[Check] q_final  (xyzw):", q_final_xyzw)

        # 7) Ensure robot is at the final IK solution
        print("\n[Exec] Sending final IK solution to robot ...")
        q_final_cmd = np.concatenate([q_head_iter, q_arm_iter])
        robot.set_positions(q_final_cmd)
        dt = robot.wait_until_reached(
            q_final_cmd,
            active_joint_indices=range(n_joints),
            pos_tol_deg=5.0,
            vel_tol_deg_s=5.0,
        )
        if dt is not None:
            print(f"[Exec] Reached final target in {dt:.2f} s.")
        else:
            print("[Exec] Final target NOT reached (timeout or no progress).")

        time.sleep(1.0)
        print("\n[Done] IK example finished.")


if __name__ == "__main__":
    main()
