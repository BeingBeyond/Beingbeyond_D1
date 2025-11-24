"""
示例 5：末端位姿 IK 求解并在真机上迭代执行。

本脚本功能：
  - 使用 D1Kinematics 读取当前末端执行器（EE）的位姿
  - 在 base 坐标系中构造一个目标末端位姿
  - 在“头部固定”的前提下，仅对机械臂进行迭代 IK 求解
  - 将每一次中间 IK 解发送到真实的 D1 头 + 臂硬件

使用场景：
  - 验证运动学模型是否与真实硬件一致
  - 检查 IK 解是否可行且收敛稳定
  - 观察小的笛卡尔空间偏移如何映射到关节空间运动。
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
    print("\033[91m警告：请始终将实体急停按钮保持在触手可及的位置。\033[0m")
    print("\033[91m      一旦机器人运动异常，请立即按下急停按钮。\033[0m\n")

    from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path
    urdf = get_default_urdf_path()
    dev = "/dev/ttyUSB0"

    # 1) 构建运动学求解器
    kin_cfg = D1KinematicsConfig(urdf_path=urdf)
    kin = D1Kinematics(kin_cfg)

    # 2) 连接真实的头 + 臂机器人
    with HeadArmRobot(urdf_path=urdf, dev=dev, baudrate=1_000_000) as robot:
        joint_names = robot.joint_names
        n_joints = len(joint_names)
        print("[信息] 关节顺序:", joint_names)

        # 2.1) 设置一个初始姿态
        q_init_deg = [
            0.0, 0.0,          # head_yaw, head_pitch
            0.0, -60.0, 60.0,  # 肩部 / 肘部
            0.0, 0.0, 0.0,     # 手腕关节
        ]
        q_init_rad = deg_list_to_rad(q_init_deg)
        robot.set_positions(q_init_rad)
        robot.wait_until_reached(q_init_rad, active_joint_indices=range(n_joints))
        time.sleep(0.5)

        # 3) 读取当前关节角，并计算当前末端位姿
        q_current = np.asarray(robot.get_positions(), dtype=float)  # shape (8,)
        q_head, q_arm = kin.split_q(q_current)

        T_base_ee = kin.ee_in_base(q_head, q_arm)
        print("[信息] 当前末端在 base 坐标系下的齐次变换矩阵:\n", T_base_ee)

        # 4) 在 base 坐标系下定义一个目标位姿（位置 + 姿态）
        p_target = np.array([0.2, 0.1, 0.2], dtype=float)
        rpy_target = np.array([0.0, np.pi / 2, 0.0], dtype=float)
        q_target_xyzw = R.from_euler("xyz", rpy_target).as_quat()  # SciPy: [x, y, z, w]

        target_quatpose = np.zeros(7, dtype=float)
        target_quatpose[0:3] = p_target
        target_quatpose[3:7] = q_target_xyzw  # D1Kinematics 按 xyzw 解释

        # 5) IK 迭代求解
        max_outer_iters = 30
        pos_tol = 0.02                     # 位置误差容忍度（米）
        rot_tol = np.deg2rad(10.0)         # 姿态误差容忍度（弧度）

        q_head_iter = q_head.copy()
        q_arm_iter = q_arm.copy()

        print("\n[IK] 开始迭代 IK 求解 ...")
        for outer in range(max_outer_iters):
            # 5.1) 用当前关节解做一次正向运动学
            T_cur = kin.ee_in_base(q_head_iter, q_arm_iter)
            p_cur = T_cur[:3, 3]
            R_cur = T_cur[:3, :3]
            q_cur_xyzw = R.from_matrix(R_cur).as_quat()

            # 5.2) 计算位姿误差
            dp = p_target - p_cur
            pos_err = np.linalg.norm(dp)

            dR = R.from_quat(q_target_xyzw) * R.from_quat(q_cur_xyzw).inv()
            rot_err = np.linalg.norm(dR.as_rotvec())

            print(
                f"[IK][{outer:02d}] 位置误差 = {pos_err:.3f} m, "
                f"姿态误差 = {np.rad2deg(rot_err):.1f} deg"
            )

            # 5.3) 判断是否收敛
            if pos_err < pos_tol and rot_err < rot_tol:
                print("[IK] 已收敛到目标位姿。")
                break

            # 5.4) 做一次仅臂部的 IK 更新（头部保持当前姿态）
            q_head_sol, q_arm_sol, cost, inner_iters = kin.ik_ee_quatpose_with_arm_only(
                target_ee_quatpose=target_quatpose,
                q_head=q_head_iter,
                q_arm_init=q_arm_iter,
            )
            print(
                f"[IK][{outer:02d}] 当前步 cost = {cost:.6f}, "
                f"内层迭代次数 = {inner_iters}"
            )
            print("[IK] 头部关节（rad）:", q_head_sol)
            print("[IK] 机械臂关节（deg）:", rad_list_to_deg(q_arm_sol.tolist()))

            # 5.5) 更新当前解
            q_head_iter = q_head_sol
            q_arm_iter = q_arm_sol

            # 5.6) 将本次解发送给真实机器人
            q_cmd = np.concatenate([q_head_iter, q_arm_iter])
            robot.set_positions(q_cmd)

        else:
            print("[IK] 警告：达到最大外层迭代次数，仍未满足收敛条件。")

        # 6) 做一次最终的 FK 检查
        T_final = kin.ee_in_base(q_head_iter, q_arm_iter)
        p_final = T_final[:3, 3]
        R_final = T_final[:3, :3]
        q_final_xyzw = R.from_matrix(R_final).as_quat()

        print("\n[检查] 目标位置 p_target:", p_target)
        print("[检查] 最终位置 p_final :", p_final)
        print("[检查] 目标姿态四元数 (xyzw):", q_target_xyzw)
        print("[检查] 最终姿态四元数 (xyzw):", q_final_xyzw)

        # 7) 确保机器人已经到达最终 IK 解
        print("\n[执行] 将最终 IK 解发送给机器人 ...")
        q_final_cmd = np.concatenate([q_head_iter, q_arm_iter])
        robot.set_positions(q_final_cmd)
        dt = robot.wait_until_reached(
            q_final_cmd,
            active_joint_indices=range(n_joints),
            pos_tol_deg=5.0,
            vel_tol_deg_s=5.0,
        )
        if dt is not None:
            print(f"[执行] 已在 {dt:.2f} s 内到达最终目标位姿。")
        else:
            print("[执行] 未能在限定时间内到达最终目标（超时或无明显进展）。")

        time.sleep(1.0)
        print("\n[完成] IK 示例结束。")


if __name__ == "__main__":
    main()