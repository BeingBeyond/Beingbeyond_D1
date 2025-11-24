"""
RealSense camera helper for D1 examples.

This module defines the RealSenseCamera class, which:
  - Configures RGB and depth streams at a given resolution and frame rate
  - Provides get_aligned_frames() to return aligned (color, depth) arrays
  - Supports optional depth filtering (spatial, temporal, hole filling, etc.)
  - Exposes camera intrinsics for color and depth streams

Use this class in examples to keep RealSense setup and frame handling
separate from the main robot control logic.
"""
import numpy as np
import pyrealsense2 as rs
import time


class RealSenseCamera:
    def __init__(self, width=640, height=480, hz=30):
        self.width = width
        self.height = height
        self._started = False

        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, hz)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, hz)

        self.pipeline = rs.pipeline()
        wrapper = rs.pipeline_wrapper(self.pipeline)
        resolved = self.config.resolve(wrapper)
        self.device = resolved.get_device()
        self.product_line = self.device.get_info(rs.camera_info.product_line)
        self.depth_sensor = self.device.first_depth_sensor()

        self.profile = self.pipeline.start(self.config)
        self._started = True

        self.align = rs.align(rs.stream.color)
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.decimation = rs.decimation_filter()
        self.hole_filling = rs.hole_filling_filter()
        self.threshold = rs.threshold_filter()

    def _filter_depth_frame(self, depth_frame: rs.frame) -> rs.frame:
        f = self.spatial.process(depth_frame)
        f = self.temporal.process(f)
        f = self.hole_filling.process(f)
        f = self.decimation.process(f)
        f = self.threshold.process(f)
        return f

    def get_aligned_frames(self, raw=False, filtered=False):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if filtered:
            depth_frame = self._filter_depth_frame(depth_frame)

        if raw:
            return color_frame, depth_frame

        color_np = np.asanyarray(color_frame.get_data())
        depth_m = np.float32(np.asanyarray(depth_frame.get_data())) / 1000.0
        return color_np, depth_m

    def get_camera_intrinsics(self):
        color_intrin = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        return color_intrin

    def get_camera_intrinsics_matrix(self):
        color_intrin = self.get_camera_intrinsics()
        intrinsics_matrix = np.array([[color_intrin.fx, 0, color_intrin.ppx],
                                      [0, color_intrin.fy, color_intrin.ppy],
                                      [0, 0, 1]])
        return intrinsics_matrix

    def get_camera_intrinsics_depth(self):
        depth_intrin = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        return depth_intrin

    def reset(self):
        try:
            self.device.hardware_reset()
        except Exception:
            pass
        time.sleep(0.1)

    def stop(self):
        if not getattr(self, "_started", False):
            return
        try:
            self.pipeline.stop()
        except Exception:
            pass
        self._started = False

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()