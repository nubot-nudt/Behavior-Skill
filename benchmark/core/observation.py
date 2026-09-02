"""Observation handling: policy obs preprocessing + eval environment config generation."""

import numpy as np
import torch as th
import omnigibson.utils.transform_utils as T
from omnigibson.learning.utils.eval_utils import (
    ROBOT_CAMERA_NAMES,
    TASK_NAMES_TO_INDICES,
    flatten_obs_dict,
)
from gello.robots.sim_robot.og_teleop_utils import (
        load_available_tasks,
        generate_robot_config,
    )
from omnigibson.learning.utils.eval_utils import generate_basic_environment_config
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES


def preprocess_obs(obs: dict, robot, env) -> dict:
    """Convert raw env obs to the policy format (consistent with eval.py's _preprocess_obs)."""
    obs = flatten_obs_dict(obs)
    base_pose = robot.get_position_orientation()
    cam_rel_poses = []
    for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
        camera = robot.sensors[camera_name.split("::")[1]]
        direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
        if np.allclose(direct_cam_pose, np.zeros(16)):
            cam_rel_poses.append(
                th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose))
            )
        else:
            cam_pose = T.mat2pose(
                th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32)
            )
            cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
    obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
    task_name = env.task.activity_name
    obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[task_name]], dtype=th.int64)
    return obs


def build_env_cfg(task_name: str) -> dict:
    """Build the og.Environment config (aligned with the full-task eval.py environment config)."""
    available_tasks = load_available_tasks()
    task_info = available_tasks[task_name][0]
    env_cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_info)
    env_cfg["robots"] = [generate_robot_config(task_name=task_name, task_cfg=task_info)]
    env_cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
    # Aligned with eval.py: explicitly specify the proprio_obs components so the
    # policy observation dimensions exactly match training (256-dim).
    env_cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
    env_cfg["task"]["include_obs"] = False
    return env_cfg
