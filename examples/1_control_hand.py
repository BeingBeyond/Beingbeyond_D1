"""
Example 1: Basic control of the DexHand.

This script:
    - Connects to the DexHand over CAN
    - Optionally runs a short "gesture dance" for visual inspection
    - Commands several hand poses using normalized joint values
    - Sets different joint speed and torque limits
    - Prints the current normalized joint positions read from the hardware

Use this example to verify that:
    - The hand is wired correctly
    - The CAN interface is configured (e.g. 'can0' at 1 Mbps)
    - The hand responds correctly to high-level normalized joint commands.
    - The hand motion behavior under different speed and torque settings.
"""
import time
from beingbeyond_d1_sdk.dex_hand import DexHand

def main():
    print("\033[91mWARNING: Always keep the physical emergency stop button within reach,\033[0m")
    print("\033[91m         and press it immediately if the robot motion looks unsafe.\033[0m\n")

    right_hand = DexHand(hand_type="right", can_iface="can0")

    try:
        right_hand.gesture_dance(duration_s=3, m=0.1, interval_s=0.1)

        right_hand.open_hand()

        right_hand.set_speed(speed=[1, 1, 1, 1, 1, 1])
        right_hand.set_torque(torque=[1, 1, 1, 1, 1, 1])

        right_hand.set_joint_pos([0.55, 0.8, 0.42, 0.45, 0, 0])
        time.sleep(1)
        right_hand.open_hand()
        time.sleep(1)

        print(right_hand.get_torque())
        print(right_hand.get_speed())

        right_hand.set_speed(speed=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        right_hand.set_torque(torque=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        right_hand.set_joint_pos([0.55, 0.8, 0.42, 0.45, 0, 0])
        time.sleep(5)

        right_hand.set_speed(speed=[1, 1, 1, 1, 1, 1])
        right_hand.set_torque(torque=[1, 1, 1, 1, 1, 1])

        print(right_hand.read_joint_pos())
        time.sleep(1)
        right_hand.half_close_hand()
        print(right_hand.read_joint_pos())

        # right.open_hand()


    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")

    finally:
        try:
            safe_open = [0.0] * 6
            right_hand.set_joint_pos(safe_open)
            time.sleep(0.1)
        except Exception as e:
            print(f"Failed to send safe-open: {e}")
        try:
            right_hand.close_can()
        except Exception as e:
            print(f"Failed to close: {e}")

if __name__ == '__main__':
    main()