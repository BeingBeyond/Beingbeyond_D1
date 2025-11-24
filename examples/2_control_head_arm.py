"""
Example 2: Basic motion test for the head + arm.

This script:
  - Connects to the head + arm controller
  - Moves all joints to 0 degrees
  - Then, for each joint in order:
      * Prints the current joint angles and velocities
      * Moves that joint to +15 degrees, then -15 degrees, then back to 0
  - Uses wait_until_reached() to block until each small motion is completed

Use this example to:
  - Verify that all head/arm joints are connected and respond
  - Check joint ordering and direction (sign of positive motion)
  - Confirm basic motion without running complex trajectories.
"""
import time
from beingbeyond_d1_sdk import HeadArmRobot

from utils import deg_list_to_rad, rad_list_to_deg


def main():
    print("\033[91mWARNING: Always keep the physical emergency stop button within reach,\033[0m")
    print("\033[91m         and press it immediately if the robot motion looks unsafe.\033[0m\n")

    from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path
    urdf = get_default_urdf_path()
    dev = "/dev/ttyUSB0"

    with HeadArmRobot(urdf_path=urdf, dev=dev) as robot:
        joint_names = robot.joint_names
        n_joints = len(joint_names)
        print("Joint order:", joint_names)

        # 1. set all joints to 0 position
        q_init_deg = [0.0] * n_joints
        q_init_rad = deg_list_to_rad(q_init_deg)
        print("\n[Step] Move all joints to 0 deg")
        robot.set_positions(q_init_rad)
        # blocking wait until reached
        robot.wait_until_reached(q_init_rad, active_joint_indices=range(n_joints))
        time.sleep(0.5)

        # 2. for each joint, move +15 deg, -15 deg, back to 0 deg
        for idx, name in enumerate(joint_names):
            print(f"\n====== Joint {idx}: {name} ======")

            # get current state
            q_rad, dq_rad = robot.get_positions_and_velocities()
            q_deg = rad_list_to_deg(q_rad)
            dq_deg = rad_list_to_deg(dq_rad)

            print(f"  Current q (deg): {q_deg}")
            print(f"  Current dq (deg/s) approx: {dq_deg}")
            print(f"  This joint q[{idx}] = {q_deg[idx]:.2f} deg, dq[{idx}] = {dq_deg[idx]:.2f} deg/s")


            base_deg = [0.0] * n_joints

            # 2.1 +15 deg
            target_deg = base_deg.copy()
            target_deg[idx] = 15.0
            target_rad = deg_list_to_rad(target_deg)
            print(f"  Move joint {idx} ({name}) to +15 deg")
            robot.set_positions(target_rad)
            robot.wait_until_reached(target_rad, active_joint_indices=[idx])

            # 2.2 -15 deg
            target_deg[idx] = -15.0
            target_rad = deg_list_to_rad(target_deg)
            print(f"  Move joint {idx} ({name}) to -15 deg")
            robot.set_positions(target_rad)
            robot.wait_until_reached(target_rad, active_joint_indices=[idx])

            # 2.3 back to 0 deg
            target_deg[idx] = 0.0
            target_rad = deg_list_to_rad(target_deg)
            print(f"  Move joint {idx} ({name}) back to 0 deg")
            robot.set_positions(target_rad)
            robot.wait_until_reached(target_rad, active_joint_indices=[idx])

        print("\n[Done] All joints tested (+15 / -15 / 0).")

if __name__ == "__main__":
    main()