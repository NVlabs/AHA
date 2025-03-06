# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from pyrep.const import RenderMode
from tqdm import tqdm

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.waypoints import Waypoint
from rlbench.environment import DIR_PATH, Environment
from rlbench.task_environment import TaskEnvironment
from rlbench.observation_config import ObservationConfig

from failgen.utils import name_to_class


class RLBenchContext:
    env: Environment
    task_env: TaskEnvironment
    frames: Dict[str, List[np.ndarray]]

    def __init__(self, env: Environment, task_env: TaskEnvironment):
        self.env = env
        self.task_env = task_env
        self.frames = dict(
            front=[],
            overhead=[],
            left_shoulder=[],
            right_shoulder=[],
        )

    def save_frames(self, save_folder: Path) -> None:
        for cam_name in self.frames.keys():
            for idx, frame in enumerate(self.frames[cam_name]):
                pil_image = Image.fromarray(frame)
                pil_image.save(save_folder / f"{cam_name}_{idx}.png")

    def reset(self):
        self.frames = dict(
            front=[],
            overhead=[],
            left_shoulder=[],
            right_shoulder=[],
        )


rlbench_ctx: Optional[RLBenchContext] = None


def on_waypoint(_: Waypoint) -> None:
    global rlbench_ctx

    if rlbench_ctx is None:
        return

    obs = rlbench_ctx.task_env.get_observation()
    rlbench_ctx.frames["front"].append(obs.front_rgb)
    rlbench_ctx.frames["overhead"].append(obs.overhead_rgb)
    rlbench_ctx.frames["left_shoulder"].append(obs.left_shoulder_rgb)
    rlbench_ctx.frames["right_shoulder"].append(obs.right_shoulder_rgb)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="basketball_in_hoop",
        help="The task to use for the data generation process",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="The number of episodes to run the data-generator for",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Whether or not to run in headless mode",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=10,
        help="The maximum number of times to try collecting a single demo",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="/home/gregor/data/keyframe_data",
        help="The location where to save the collected keyframes",
    )

    global rlbench_ctx

    args = parser.parse_args()

    task_name = args.task
    task_folder = (Path(DIR_PATH) / "tasks").resolve()

    print(f"Collecting data from task {task_name}")

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.front_camera.rgb = True
    obs_config.front_camera.image_size = [256, 256]
    obs_config.front_camera.render_mode = RenderMode.OPENGL3
    obs_config.overhead_camera.rgb = True
    obs_config.overhead_camera.image_size = [256, 256]
    obs_config.overhead_camera.render_mode = RenderMode.OPENGL3
    obs_config.left_shoulder_camera.rgb = True
    obs_config.left_shoulder_camera.image_size = [256, 256]
    obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL3
    obs_config.right_shoulder_camera.rgb = True
    obs_config.right_shoulder_camera.image_size = [256, 256]
    obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL3

    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
        ),
        obs_config=obs_config,
        headless=args.headless,
    )
    env.launch()

    task_class = name_to_class(task_name, str(task_folder))
    if task_class is None:
        raise RuntimeError(f"Couldn't instantiate task '{task_name}'")
    task_env = env.get_task(task_class)

    rlbench_ctx = RLBenchContext(env, task_env)

    for ep_idx in tqdm(range(args.num_episodes)):
        rlbench_ctx.reset()
        (_,) = task_env.get_demos(
            amount=1,
            live_demos=True,
            max_attempts=args.max_tries,
            callable_each_waypoint=on_waypoint,
        )
        save_folder = Path(args.output_folder) / task_name / f"episode_{ep_idx}"
        save_folder.mkdir(parents=True, exist_ok=True)
        rlbench_ctx.save_frames(save_folder)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
