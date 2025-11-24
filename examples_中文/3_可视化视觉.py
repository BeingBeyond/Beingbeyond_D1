"""
示例 3：RealSense RGB-D 可视化。

本脚本功能：
  - 使用 RealSenseCamera 辅助类打开 RealSense 相机
  - 实时获取对齐后的 RGB 与深度图
  - 将 RGB 从 RGB 转为 BGR（OpenCV 显示）
  - 将深度图归一化后做伪彩色显示
  - 将 RGB 与深度伪彩色拼接到同一个 OpenCV 窗口中

在 OpenCV 窗口中按下 'q' 键退出。

使用场景：
  - 检查是否成功识别到 RealSense 相机
  - 检查 RGB 与深度对齐是否正常
  - 对比有无滤波时的深度质量。
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