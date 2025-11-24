import threading
import traceback

import numpy as np

from beingbeyond_d1_sdk import HeadArmRobot
from beingbeyond_d1_sdk import DexHand

from vision import RealSenseCamera

class D1Robot:
    def __init__(self, urdf_path: str, arm_dev: str, arm_baud: int,
                 hand_type: str, hand_can: str, hand_baud: int):
        self.head_arm = HeadArmRobot(
            urdf_path=urdf_path,
            dev=arm_dev,
            baudrate=arm_baud,
        )
        self.hand = DexHand(
            hand_type=hand_type,
            can_iface=hand_can,
            baudrate=hand_baud,
        )

        self.vision = RealSenseCamera()

        # Vision thread control
        self._vision_thread = None
        self._vision_stop_event = None

    def start_vision_thread(
        self,
        filtered: bool = False,
        window_name: str = "RealSense RGB | Depth (D1 View)",
    ):
        """
        Start a background thread that continuously displays RGB + Depth.
        Press 'q' in the window to stop the vision thread.
        """
        if self._vision_thread is not None and self._vision_thread.is_alive():
            return  # already running

        self._vision_stop_event = threading.Event()

        def _vision_loop():
            import cv2

            try:
                with self.vision as camera:
                    while not self._vision_stop_event.is_set():
                        color_rgb, depth_m = camera.get_aligned_frames(filtered=filtered)

                        # RGB: convert to BGR for OpenCV
                        color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)

                        # Depth: normalize and apply colormap
                        depth_valid = np.nan_to_num(
                            depth_m,
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                        depth_norm = cv2.normalize(depth_valid,None,0,255,cv2.NORM_MINMAX,)
                        depth_u8 = depth_norm.astype(np.uint8)
                        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

                        # Ensure shapes match
                        if depth_color.shape[:2] != color_bgr.shape[:2]:
                            depth_color = cv2.resize(
                                depth_color,
                                (color_bgr.shape[1], color_bgr.shape[0]),
                            )

                        vis = np.hstack([color_bgr, depth_color])
                        cv2.imshow(window_name, vis)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            self._vision_stop_event.set()
                            break

            except Exception as e:
                print("[VisionThread] Error:")
                print(e)
                traceback.print_exc()
            finally:
                try:
                    import cv2
                    cv2.destroyWindow(window_name)
                except Exception:
                    pass

        self._vision_thread = threading.Thread(
            target=_vision_loop,
            daemon=True,
        )
        self._vision_thread.start()

    def stop_vision_thread(self, join_timeout: float = 2.0):
        """
        Request the vision thread to stop and join it.
        """
        if self._vision_stop_event is not None:
            self._vision_stop_event.set()
        if self._vision_thread is not None and self._vision_thread.is_alive():
            self._vision_thread.join(join_timeout)

    def set_q(self, q):
        self.head_arm.set_positions(q[:8])
        self.hand.set_joint_pos(q[8:])

    def get_q(self):
        q_arm = self.head_arm.get_positions()
        q_hand = self.hand.read_joint_pos()
        return q_arm + q_hand

    def close(self):
        # Stop vision thread first
        try:
            self.stop_vision_thread()
        except Exception:
            pass

        # Then release robot hardware
        try:
            self.hand.open_hand()
            self.hand.close_can()
        except Exception:
            pass
        try:
            self.head_arm.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()