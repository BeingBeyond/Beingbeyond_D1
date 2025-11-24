"""
Example 3: RealSense RGB-D viewer.

This script:
  - Opens a RealSense camera using the RealSenseCamera helper
  - Streams aligned RGB and depth frames
  - Converts RGB to BGR and visualizes depth as a color map
  - Displays RGB and depth side by side in a single OpenCV window

Press 'q' in the OpenCV window to exit.

Use this example to:
  - Verify that the RealSense camera is detected
  - Check that RGB and depth alignment works
  - Inspect the depth quality with and without filtering.
"""
import numpy as np

from vision import RealSenseCamera


def main():
    import cv2

    with RealSenseCamera() as camera:
        while True:
            color_rgb, depth_m = camera.get_aligned_frames(filtered=False)

            color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)

            depth_valid = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
            depth_norm = cv2.normalize(depth_valid, None, 0, 255, cv2.NORM_MINMAX)
            depth_u8 = depth_norm.astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

            if depth_color.shape[:2] != color_bgr.shape[:2]:
                depth_color = cv2.resize(depth_color, (color_bgr.shape[1], color_bgr.shape[0]))

            vis = np.hstack([color_bgr, depth_color])
            cv2.imshow("RealSense RGB | Depth (filtered)", vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
