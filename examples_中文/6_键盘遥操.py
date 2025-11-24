"""
示例 6：键盘控制 D1 末端执行器（EE）在笛卡尔空间中的实时遥操作。

功能特点：
  - 使用 D1Kinematics，通过键盘在笛卡尔空间控制末端执行器
    （键位：w/s/a/d/z/x/u/o/i/k/j/l）
  - 在线求解 IK（仅机械臂，头部保持当前姿态）
  - 将每一步 IK 解发送给真实 D1 机器人（头 + 臂 + 手）
  - 空格键 [Space] 在 hand_pos = 0 / 1 之间切换，并通过 map_hand(hand_pos)
    映射为 6 维手部关节归一化值

使用方法：
  - 在终端中运行本脚本，并确保实体急停按钮就在手边
  - 键位说明：
      w / s : X+ / X-
      a / d : Y+ / Y-
      z / x : Z+ / Z-
      u / o : roll  + / -
      i / k : pitch + / -
      j / l : yaw   + / -
      空格  : 在 0 / 1 之间切换 hand_pos（开/合），并映射到灵巧手参数
      r     : 将目标 EE 位姿与关节配置重置为初始状态
      h     : 打印帮助信息
      q     : 退出
"""
import math
import sys
import time
import select
import termios
import tty

import numpy as np

from beingbeyond_d1_sdk.pin_kinematics import (
    D1Kinematics,
    D1KinematicsConfig,
    IKOptions,
)
from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path
from D1_robot import D1Robot


def deg_list_to_rad(xs):
    return [math.radians(v) for v in xs]


def rad_list_to_deg(xs):
    return [math.degrees(v) for v in xs]


def _getch_nonblocking(timeout: float):
    dr, _, _ = select.select([sys.stdin], [], [], timeout)
    if not dr:
        return None
    ch = sys.stdin.read(1)
    return ch

def _set_terminal_raw():
    if not sys.stdin.isatty():
        print(
            "stdin 不是一个 TTY，跳过原始模式配置。\n"
            "可能正在 IDE 中运行。\n"
            "将使用行输入模式（回车后生效）。"
        )
        return None, None
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return fd, old_settings

def _restore_terminal(fd, old_settings):
    if fd is None or old_settings is None:
        return
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def _rot_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )

def _rot_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )

def _rot_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

def _orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt



def map_hand(hand_pos: float):
    A = [0.64, 0.8, 0.54, 0.58, 0.0, 0.0]  # hand_pos = 1
    B = [0.0, 0.8, 0.0, 0.0, 0.0, 0.0]     # hand_pos = 0
    t = 0.0 if hand_pos < 0.0 else 1.0 if hand_pos > 1.0 else hand_pos
    return [b + t * (a - b) for a, b in zip(A, B)]



class EETeleopController:
    def __init__(
        self,
        p0: np.ndarray,
        R0: np.ndarray,
        q_head0: np.ndarray,
        q_arm0: np.ndarray,
        pos_step: float,
        ori_step: float,
        max_offset: np.ndarray,
    ) -> None:
        self.p0 = p0.copy()
        self.R0 = R0.copy()
        self.p_des = p0.copy()
        self.R_des = R0.copy()

        self.q_head = q_head0.copy()
        self.q_arm = q_arm0.copy()

        self.pos_step = float(pos_step)
        self.ori_step = float(ori_step)
        self.max_offset = np.asarray(max_offset, dtype=float).reshape(3)

        # toggle_flag 作为 hand_pos 的标志位（0 或 1）
        self.toggle_flag = 0.0

        self._help_template = """
键盘遥操作（末端在 base 坐标系下）

  平移:
    w / s : X+ / X-
    a / d : Y+ / Y-
    z / x : Z+ / Z-

  姿态:
    u / o : roll  + / -
    i / k : pitch + / -
    j / l : yaw   + / -

  其他:
    [Space] : 在 hand_pos = 0 与 1 之间切换，当前 = {flag}
    r       : 将 EE 目标位姿与关节配置重置为初始状态
    h       : 打印此帮助信息
    q       : 退出
"""

    def print_help(self):
        print(self._help_template.format(flag=self.toggle_flag))

    def _clamp_workspace(self):
        offset = self.p_des - self.p0
        offset = np.clip(offset, -self.max_offset, self.max_offset)
        self.p_des = self.p0 + offset

    def handle_key(self, ch: str, q_head_zero: np.ndarray, q_arm_zero: np.ndarray):
        if ch == "q":
            print("收到退出遥操作请求（q）。")
            return "quit"
        elif ch == "h":
            self.print_help()
            return None

        # translation
        if ch == "w":
            self.p_des[0] += self.pos_step
        elif ch == "s":
            self.p_des[0] -= self.pos_step
        elif ch == "a":
            self.p_des[1] += self.pos_step
        elif ch == "d":
            self.p_des[1] -= self.pos_step
        elif ch == "z":
            self.p_des[2] += self.pos_step
        elif ch == "x":
            self.p_des[2] -= self.pos_step

        # orientation
        elif ch == "u":
            dR = _rot_x(self.ori_step)
            self.R_des = _orthonormalize(dR @ self.R_des)
        elif ch == "o":
            dR = _rot_x(-self.ori_step)
            self.R_des = _orthonormalize(dR @ self.R_des)
        elif ch == "i":
            dR = _rot_y(self.ori_step)
            self.R_des = _orthonormalize(dR @ self.R_des)
        elif ch == "k":
            dR = _rot_y(-self.ori_step)
            self.R_des = _orthonormalize(dR @ self.R_des)
        elif ch == "j":
            dR = _rot_z(self.ori_step)
            self.R_des = _orthonormalize(dR @ self.R_des)
        elif ch == "l":
            dR = _rot_z(-self.ori_step)
            self.R_des = _orthonormalize(dR @ self.R_des)

        # Space: hand_pos 0 <-> 1
        elif ch == " ":
            self.toggle_flag = 1.0 - self.toggle_flag
            print(f"[Space] hand_pos 已切换为 {self.toggle_flag}")

        # reset
        elif ch == "r":
            self.p_des = self.p0.copy()
            self.R_des = self.R0.copy()
            self.q_head = q_head_zero.copy()
            self.q_arm = q_arm_zero.copy()
            print("已将 EE 目标位姿与关节状态重置为初始值。")

        self._clamp_workspace()
        return None

    def build_target_T(self) -> np.ndarray:
        T = np.eye(4, dtype=float)
        T[:3, :3] = self.R_des
        T[:3, 3] = self.p_des
        return T

    def set_q(self, q_head: np.ndarray, q_arm: np.ndarray):
        self.q_head = np.asarray(q_head, dtype=float).copy()
        self.q_arm = np.asarray(q_arm, dtype=float).copy()




def solve_ik_safe(
    kin: D1Kinematics,
    target_T: np.ndarray,
    q_head: np.ndarray,
    q_arm: np.ndarray,
    err_thr: float,
):
    q_head_out, q_arm_out, err, iters = kin.ik_T_ee_with_arm_only(
        target_T_base_ee=target_T,
        q_head=q_head,
        q_arm_init=q_arm,
    )
    if np.isnan(err) or err > err_thr:
        return q_head, q_arm, err, iters, False
    return q_head_out, q_arm_out, err, iters, True



def main() -> int:
    print("\033[91m警告：请始终将实体急停按钮保持在触手可及的位置。\033[0m")
    print("\033[91m      一旦机器人运动异常，请立即按下急停按钮。\033[0m\n")

    urdf = get_default_urdf_path()

    kin_cfg = D1KinematicsConfig(urdf_path=urdf)
    kin = D1Kinematics(kin_cfg)


    try:
        with D1Robot(
            urdf_path=urdf,
            arm_dev="/dev/ttyUSB0",
            arm_baud=1_000_000,
            hand_type="right",
            hand_can="can0",
            hand_baud=1_000_000,
        ) as robot:

            q_init_deg = [
                0.0, 0.0,
                0.0, -60.0, 60.0, 0.0, 0.0, 0.0,
            ]
            q_init_rad = deg_list_to_rad(q_init_deg)
            robot.head_arm.set_positions(q_init_rad)
            robot.head_arm.wait_until_reached(
                q_init_rad,
                active_joint_indices=range(len(q_init_rad)),
            )
            time.sleep(0.5)


            q_headarm = np.asarray(robot.head_arm.get_positions(), dtype=float)
            q_head, q_arm = kin.split_q(q_headarm)


            T0 = kin.ee_in_base(q_head, q_arm)
            R0 = T0[:3, :3]
            p0 = T0[:3, 3].copy()

            pos_step = 0.01  # m
            ori_step = 5.0 * math.pi / 180.0  # rad
            max_offset = np.array([0.5, 0.5, 0.5], dtype=float)
            ik_fail_thr = 0.05  # m

            controller = EETeleopController(
                p0=p0,
                R0=R0,
                q_head0=q_head,
                q_arm0=q_arm,
                pos_step=pos_step,
                ori_step=ori_step,
                max_offset=max_offset,
            )

            controller.print_help()

            use_raw = sys.stdin.isatty()
            fd, old_settings = _set_terminal_raw()
            print("进入键盘遥操作循环，按下 'q' 退出。")

            dt = 0.01

            try:
                while True:
                    if use_raw:
                        ch = _getch_nonblocking(timeout=dt)
                    else:
                        try:
                            line = input(
                                "请输入指令 [w/s/a/d/z/x/u/o/i/k/j/l/space/r/h/q]（直接回车跳过本轮）: "
                            )
                        except EOFError:
                            print("检测到 stdin EOF，退出遥操作。")
                            break
                        ch = line[0] if line else None

                    if ch is not None:
                        res = controller.handle_key(
                            ch,
                            q_head_zero=q_head,
                            q_arm_zero=q_arm,
                        )
                        if res == "quit":
                            break


                    target_T = controller.build_target_T()
                    q_head_new, q_arm_new, err, iters, ok = solve_ik_safe(
                        kin=kin,
                        target_T=target_T,
                        q_head=controller.q_head,
                        q_arm=controller.q_arm,
                        err_thr=ik_fail_thr,
                    )
                    if not ok:
                        continue

                    controller.set_q(q_head_new, q_arm_new)


                    hand_pos = float(controller.toggle_flag)
                    hand_q_norm = map_hand(hand_pos)  # 6 维


                    q_headarm_cmd = np.concatenate(
                        [controller.q_head, controller.q_arm]
                    )
                    q_cmd = list(q_headarm_cmd) + hand_q_norm


                    robot.set_q(q_cmd)

                    time.sleep(dt)

            finally:
                _restore_terminal(fd, old_settings)
                print("遥操作循环结束，终端设置已恢复。")

    except KeyboardInterrupt:
        print("\n用户中断（Ctrl+C）。")

    print("键盘遥操作示例结束。")
    return 0


if __name__ == "__main__":
    main()