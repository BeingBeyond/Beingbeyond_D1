"""
示例 1：灵巧手的控制。

本脚本功能：
    - 通过 CAN 连接到灵巧手
    - 可选执行一段简短的手势舞，用于目视检查
    - 使用归一化关节值命令多个手部姿态
    - 设置不同的关节速度和力矩限制
    - 打印从硬件读取到的当前归一化关节位置、速度和力矩

使用场景：
    - 验证灵巧手接线是否正确
    - 检查 CAN 接口配置是否正确（例如 'can0'，波特率 1 Mbps）
    - 确认灵巧手能正确响应高层归一化关节命令
    - 观察不同速度和力矩设置下的手部运动表现
"""
import time
from beingbeyond_d1_sdk.dex_hand import DexHand


def main():
    print("\033[91m警告：请始终将实体急停按钮保持在触手可及的位置。\033[0m")
    print("\033[91m      一旦机器人运动异常，请立即按下急停按钮。\033[0m\n")

    # 连接右手，CAN 接口为 can0
    right_hand = DexHand(hand_type="right", can_iface="can0")

    try:
        # 让手先做一段“手势舞”，便于远处确认手部状态
        right_hand.gesture_dance(duration_s=3, m=0.1, interval_s=0.1)

        # 打开手
        right_hand.open_hand()

        # 设定较大的速度和力矩，做一个快速动作
        right_hand.set_speed(speed=[1, 1, 1, 1, 1, 1])
        right_hand.set_torque(torque=[1, 1, 1, 1, 1, 1])

        # 设置一个示例手势（归一化关节位置）
        right_hand.set_joint_pos([0.55, 0.8, 0.42, 0.45, 0, 0])
        time.sleep(1)
        right_hand.open_hand()
        time.sleep(1)

        print("当前关节力矩:", right_hand.get_torque())
        print("当前关节速度:", right_hand.get_speed())

        # 设置较小的速度和力矩，做一个慢动作
        right_hand.set_speed(speed=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        right_hand.set_torque(torque=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        right_hand.set_joint_pos([0.55, 0.8, 0.42, 0.45, 0, 0])
        time.sleep(5)

        # 恢复速度和力矩到 1.0
        right_hand.set_speed(speed=[1, 1, 1, 1, 1, 1])
        right_hand.set_torque(torque=[1, 1, 1, 1, 1, 1])

        print("当前关节位置（归一化）:", right_hand.read_joint_pos())
        time.sleep(1)
        right_hand.half_close_hand()
        print("半握后关节位置（归一化）:", right_hand.read_joint_pos())

    except KeyboardInterrupt:
        print("\n用户中断（Ctrl+C）。")
    except Exception as e:
        print(f"发生错误: {e}")

    finally:
        # 退出前尽量把手张开到一个安全姿态，并关闭 CAN
        try:
            safe_open = [0.0] * 6
            right_hand.set_joint_pos(safe_open)
            time.sleep(0.1)
        except Exception as e:
            print(f"发送安全张开姿态失败: {e}")
        try:
            right_hand.close_can()
        except Exception as e:
            print(f"关闭 CAN 通信失败: {e}")


if __name__ == '__main__':
    main()
