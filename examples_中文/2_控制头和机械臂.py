"""
示例 2：头部 + 机械臂的基础关节运动测试。

本脚本功能：
  - 连接头部 + 机械臂控制器
  - 首先将所有关节移动到 0 度
  - 然后依次对每个关节执行以下动作：
      * 打印当前所有关节角度和速度
      * 将该关节移动到 +15 度，再移动到 -15 度，最后回到 0 度
  - 使用 wait_until_reached() 阻塞等待每一步小动作完成

使用场景：
  - 验证所有头/臂关节是否连通并能响应命令
  - 检查关节顺序是否正确、正方向是否符合预期
  - 在不运行复杂轨迹的前提下，做一次基础运动测试。
"""
import time
from beingbeyond_d1_sdk import HeadArmRobot

from utils import deg_list_to_rad, rad_list_to_deg


def main():
    print("\033[91m警告：请始终将实体急停按钮保持在触手可及的位置，\033[0m")
    print("\033[91m      一旦机器人运动异常，请立即按下急停按钮。\033[0m\n")

    from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path
    urdf = get_default_urdf_path()
    dev = "/dev/ttyUSB0"

    with HeadArmRobot(urdf_path=urdf, dev=dev) as robot:
        joint_names = robot.joint_names
        n_joints = len(joint_names)
        print("关节顺序:", joint_names)

        # 1. 将所有关节设为 0 度
        q_init_deg = [0.0] * n_joints
        q_init_rad = deg_list_to_rad(q_init_deg)
        print("\n[步骤] 将所有关节移动到 0 度 ...")
        robot.set_positions(q_init_rad)
        robot.wait_until_reached(q_init_rad, active_joint_indices=range(n_joints))
        time.sleep(0.5)

        # 2. 依次对每个关节做 +15 / -15 / 回到 0 的测试
        for idx, name in enumerate(joint_names):
            print(f"\n====== 关节 {idx}: {name} ======")

            # 读取当前状态
            q_rad, dq_rad = robot.get_positions_and_velocities()
            q_deg = rad_list_to_deg(q_rad)
            dq_deg = rad_list_to_deg(dq_rad)

            print(f"  当前各关节角度 (deg): {q_deg}")
            print(f"  当前各关节角速度 (deg/s): {dq_deg}")
            print(
                f"  当前关节[{idx}] 角度 = {q_deg[idx]:.2f} deg, "
                f"角速度 = {dq_deg[idx]:.2f} deg/s"
            )

            base_deg = [0.0] * n_joints

            # 2.1 移动到 +15 度
            target_deg = base_deg.copy()
            target_deg[idx] = 15.0
            target_rad = deg_list_to_rad(target_deg)
            print(f"  将关节 {idx} ({name}) 移动到 +15 度")
            robot.set_positions(target_rad)
            robot.wait_until_reached(target_rad, active_joint_indices=[idx])

            # 2.2 移动到 -15 度
            target_deg[idx] = -15.0
            target_rad = deg_list_to_rad(target_deg)
            print(f"  将关节 {idx} ({name}) 移动到 -15 度")
            robot.set_positions(target_rad)
            robot.wait_until_reached(target_rad, active_joint_indices=[idx])

            # 2.3 回到 0 度
            target_deg[idx] = 0.0
            target_rad = deg_list_to_rad(target_deg)
            print(f"  将关节 {idx} ({name}) 移动回 0 度")
            robot.set_positions(target_rad)
            robot.wait_until_reached(target_rad, active_joint_indices=[idx])

        print("\n[完成] 所有关节均已完成 +15 / -15 / 回零 的测试。")


if __name__ == "__main__":
    main()