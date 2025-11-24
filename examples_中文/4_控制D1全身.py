"""
示例 4：带实时 RGB-D 视角的完整 D1 控制 Demo。

本脚本功能：
  - 创建高层封装 D1Robot（包含：头 + 臂 + 手 + RealSense 相机）
  - 启动后台线程，实时显示 RGB + 深度图
  - 依次发送多个头/臂/手的组合姿态
  - 使用 wait_until_reached() 同步等待头/臂运动完成
  - 打印当前机器人状态（关节名、位置、速度、手部归一化关节）

在 RGB-D 窗口中按下 'q' 键停止视觉线程。

使用场景：
  - 做一次集成测试，验证机械臂、头部、灵巧手和相机是否一起正常工作
  - 从相机视角观察机器人行为
  - 作为遥操作或任务级应用的起始示例。
"""
import time

from utils import deg_list_to_rad, rad_list_to_deg
from D1_robot import D1Robot


def print_state(robot: D1Robot):
    q_rad, dq_rad = robot.head_arm.get_positions_and_velocities()
    q_deg = rad_list_to_deg(q_rad)
    dq_deg = rad_list_to_deg(dq_rad)

    hand_q_norm = robot.hand.read_joint_pos()

    print("\n[机器人状态]")
    print("关节名称: ", robot.head_arm.joint_names)
    print("头+臂 关节位置 (deg):  ", ["{:+5.2f}".format(v) for v in q_deg])
    print("头+臂 关节速度 (deg/s):", ["{:+5.2f}".format(v) for v in dq_deg])
    print("手部归一化位置:        ", ["{:+5.2f}".format(v) for v in hand_q_norm])


def main() -> int:
    print("=== D1 控制演示 Demo ===")
    print("\033[91m警告：请始终将实体急停按钮保持在触手可及的位置。\033[0m")
    print("\033[91m      一旦机器人运动异常，请立即按下急停按钮。\033[0m\n")

    from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path
    urdf = get_default_urdf_path()

    try:
        with D1Robot(
            urdf_path=urdf,
            arm_dev="/dev/ttyUSB0",
            arm_baud=1000000,
            hand_type="right",
            hand_can="can0",
            hand_baud=1000000,
        ) as robot:
            # 自定义速度/加速度
            # vels_rad = deg_list_to_rad([30.0] * 8)
            # accs_rad = deg_list_to_rad([30.0] * 8)
            # robot.head_arm.set_profile(vels_rad, accs_rad)

            # 启动 RGB + 深度图显示线程
            robot.start_vision_thread(filtered=False)

            print_state(robot)

            # 1) 类 home 姿态，并打印状态
            head_q = deg_list_to_rad([0.0, 0.0])
            arm_q = deg_list_to_rad([0.0, -90.0, 90.0, 0.0, 0.0, 0.0])
            hand_q_norm = [0.55, 0.8, 0.42, 0.45, 0.0, 0.0]
            q = head_q + arm_q + hand_q_norm
            robot.set_q(q)
            print_state(robot)
            robot.head_arm.wait_until_reached(
                head_q + arm_q,
                pos_tol_deg=5.0,
                vel_tol_deg_s=5.0,
            )
            print_state(robot)

            # 2) 第一个姿态
            head_q = deg_list_to_rad([-30.0, 30.0])
            arm_q = deg_list_to_rad([30.0, -60.0, 60.0, 30.0, 30.0, 30.0])
            robot.hand.set_speed(speed=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
            hand_q_norm = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
            q = head_q + arm_q + hand_q_norm
            robot.set_q(q)
            robot.head_arm.wait_until_reached(
                head_q + arm_q,
                pos_tol_deg=5.0,
                vel_tol_deg_s=5.0,
            )
            time.sleep(3.0)

            # 3) 第二个姿态
            head_q = deg_list_to_rad([-15.0, 15.0])
            arm_q = deg_list_to_rad([0.0, -30.0, 70.0, 0.0, -30.0, 0.0])
            hand_q_norm = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
            q = head_q + arm_q + hand_q_norm
            robot.set_q(q)
            time.sleep(3.0)

            # 4) 回到一个中性姿态
            head_q = deg_list_to_rad([0.0, 0.0])
            arm_q = deg_list_to_rad([0.0, -60.0, 60.0, 0.0, 0.0, 0.0])
            robot.hand.set_speed(speed=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            hand_q_norm = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            q = head_q + arm_q + hand_q_norm
            robot.set_q(q)
            t_home = robot.head_arm.wait_until_reached(
                head_q + arm_q,
                pos_tol_deg=5.0,
                vel_tol_deg_s=5.0,
            )
            if t_home is not None:
                print(f"已在 {t_home:.2f} s 内回到中性姿态。")
            else:
                print("未能在限定时间内回到中性姿态（超时或无明显进展）。")
            time.sleep(1.0)

            # 5) 设置自定义的速度/加速度
            print("\n正在设置自定义运动轮廓 ...")
            vels_rad = deg_list_to_rad([90.0] * 8)
            accs_rad = deg_list_to_rad([90.0] * 8)
            robot.head_arm.set_profile(vels_rad, accs_rad)

            # 6) 第三个姿态
            head_q = deg_list_to_rad([-30.0, 15.0])
            arm_q = deg_list_to_rad([-15.0, -50.0, 50.0, 70.0, 30.0, 10.0])
            hand_q_norm = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
            q = head_q + arm_q + hand_q_norm
            robot.set_q(q)
            t2 = robot.head_arm.wait_until_reached(
                head_q + arm_q,
                pos_tol_deg=5.0,
                vel_tol_deg_s=5.0,
            )
            if t2 is not None:
                print(f"已在 {t2:.2f} s 内到达第三个姿态。")
            else:
                print("未能在限定时间内到达第三个姿态（超时或无明显进展）。")

            time.sleep(1.0)
            print("\n演示结束，`close()` 中将执行回 home 和资源释放 ...")

    except KeyboardInterrupt:
        print("\n用户中断（Ctrl+C）。")

    print("Demo 结束。")
    return 0


if __name__ == "__main__":
    main()