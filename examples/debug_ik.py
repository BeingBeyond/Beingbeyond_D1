import numpy as np
from scipy.spatial.transform import Rotation as R
from beingbeyond_d1_sdk.pin_kinematics import D1Kinematics, D1KinematicsConfig
from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path

urdf = get_default_urdf_path()
kin = D1Kinematics(D1KinematicsConfig(urdf_path=urdf))

q_head_iter = np.zeros(2)
q_arm_iter  = np.zeros(6)

p_target = np.array([0.2, 0.1, 0.2])
rpy_target = np.array([0.0, np.pi/2, 0.0])
q_target_xyzw = R.from_euler("xyz", rpy_target).as_quat()
target_quatpose = np.concatenate([p_target, q_target_xyzw])

for outer in range(5):
    print(f"=== outer {outer} ===")
    q_head_sol, q_arm_sol, cost, inner_iters = kin.ik_ee_quatpose_with_arm_only(
        target_ee_quatpose=target_quatpose,
        q_head=q_head_iter,
        q_arm_init=q_arm_iter,
    )
    print("  cost:", cost, "inner_iters:", inner_iters)

    # debug
    print("  head:", q_head_sol)
    print("  arm :", q_arm_sol)

    q_head_iter = q_head_sol
    q_arm_iter  = q_arm_sol
