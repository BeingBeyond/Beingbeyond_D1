"""
Example 4: Full D1 control demo with live RGB-D view.

This script:
  - Instantiates a high-level D1Robot wrapper (head + arm + hand + RealSense)
  - Starts a background thread to display RGB + depth in real time
  - Commands several head/arm/hand poses in sequence
  - Uses wait_until_reached() to synchronize arm/head motions
  - Prints the current robot state (joint names, positions, velocities, hand q)

Press 'q' in the RGB-D window to stop the vision thread.

Use this example as a high-level integration demo to:
  - Validate that the arm, head, hand and camera all work together
  - Inspect the robot behavior from the camera's point of view
  - Use as a starting point for teleoperation or task-level applications.
"""
import time

from utils import deg_list_to_rad, rad_list_to_deg
from D1_robot import D1Robot

def print_state(robot: D1Robot):
    q_rad, dq_rad = robot.head_arm.get_positions_and_velocities()
    q_deg = rad_list_to_deg(q_rad)
    dq_deg = rad_list_to_deg(dq_rad)

    hand_q_norm = robot.hand.read_joint_pos()

    print("\n[Robot State]")
    print("Joint names: ", robot.head_arm.joint_names)
    print("HeadArm Pos (deg)  :", ["{:+5.2f}".format(v) for v in q_deg])
    print("HeadArm Vel (deg/s):", ["{:+5.2f}".format(v) for v in dq_deg])
    print("Hand normed pos    :", ["{:+5.2f}".format(v) for v in hand_q_norm])


def main() -> int:
    print("=== D1 Control Demo ===")
    print("\033[91mWARNING: Always keep the physical emergency stop button within reach.\033[0m")
    print("\033[91m         Press it immediately if the robot motion looks unsafe.\033[0m\n")

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
            # vels_rad = deg_list_to_rad([30.0] * 8)
            # accs_rad = deg_list_to_rad([30.0] * 8)
            # robot.head_arm.set_profile(vels_rad, accs_rad)


            # Start the RGB+Depth display thread
            robot.start_vision_thread(filtered=False)

            print_state(robot)

            # 1) Home-like pose, then read and print state
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

            # 2) First posture
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

            # 3) Second posture
            head_q = deg_list_to_rad([-15.0, 15.0])
            arm_q = deg_list_to_rad([0.0, -30.0, 70.0, 0.0, -30.0, 0.0])
            hand_q_norm = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
            q = head_q + arm_q + hand_q_norm
            robot.set_q(q)
            time.sleep(3.0)

            # 4) Back to a neutral posture
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
                print(f"Reached neutral posture in {t_home:.2f} s.")
            else:
                print("Neutral posture NOT reached (timeout or no progress).")
            time.sleep(1.0)

            # 5) Configure a motion profile
            print("\nSetting custom motion profile ...")
            vels_rad = deg_list_to_rad([90.0] * 8)
            accs_rad = deg_list_to_rad([90.0] * 8)
            robot.head_arm.set_profile(vels_rad, accs_rad)

            # 6) Third posture
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
                print(f"Reached third posture in {t2:.2f} s.")
            else:
                print("Third posture NOT reached (timeout or no progress).")

            time.sleep(1.0)
            print("\nDemo finished, homing and releasing resources in `close()` ...")

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")

    print("Done.")
    return 0


if __name__ == "__main__":
    main()
