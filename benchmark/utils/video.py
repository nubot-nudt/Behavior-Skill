"""Video utilities: composite-frame extraction from three cameras and mp4 saving."""

import os
from pathlib import Path
import torch as _th
import cv2
import numpy as np


def _extract_video_frame(obs: dict):
    """
    Stitch the three camera RGBs into a composite frame exactly like eval.py:
        [left_wrist (224×224)]                 [          ]
        [                     ]   hstack  [head (448×448)]
        [right_wrist (224×224)]                [          ]

    The wrist cameras move with the grippers, so the gripper is stationary
    within them → removes the gripper jitter seen from the external head view.
    Degraded when any camera is missing: head only / wrist only / None.
    """

    CAM = {
        "head":        "robot_r1::robot_r1:zed_link:Camera:0::rgb",
        "left_wrist":  "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
        "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
    }

    def _to_uint8(frame):
        if isinstance(frame, _th.Tensor):
            frame = frame.cpu().numpy()
        frame = frame[..., :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0.0, 1.0)
            frame = (frame * 255).astype(np.uint8)
        return frame

    has_head  = CAM["head"]        in obs
    has_left  = CAM["left_wrist"]  in obs
    has_right = CAM["right_wrist"] in obs

    if has_head and has_left and has_right:
        # identical tiling to eval.py._write_video()
        left_rgb  = cv2.resize(_to_uint8(obs[CAM["left_wrist"]]),  (224, 224))
        right_rgb = cv2.resize(_to_uint8(obs[CAM["right_wrist"]]), (224, 224))
        head_rgb  = cv2.resize(_to_uint8(obs[CAM["head"]]),        (448, 448))
        return np.hstack([np.vstack([left_rgb, right_rgb]), head_rgb])
    elif has_head:
        return _to_uint8(obs[CAM["head"]])
    elif has_left:
        return _to_uint8(obs[CAM["left_wrist"]])
    elif has_right:
        return _to_uint8(obs[CAM["right_wrist"]])
    return None


def _save_video(frames: list, video_path: str, fps: int = 15) -> None:
    """Write a list of RGB uint8 frames to an mp4 file.
    Prefers PyAV (same as the official eval.py, libx264/yuv420p), falls back to cv2 mp4v.
    """
    if not frames:
        return
    os.makedirs(Path(video_path).parent, exist_ok=True)
    arr = np.stack(frames)  # (N, H, W, 3)
    try:
        from omnigibson.learning.utils.obs_utils import create_video_writer, write_video
        h, w = frames[0].shape[:2]
        writer = create_video_writer(fpath=video_path, resolution=(h, w), rate=fps)
        write_video(arr, video_writer=writer)
        container, _ = writer
        container.close()
    except Exception:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        for frame in frames:
            writer.write(frame[..., ::-1])  # RGB → BGR
        writer.release()
