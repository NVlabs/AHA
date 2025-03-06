# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from typing import List, Optional

import numpy as np
from moviepy.editor import ImageSequenceClip
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.observation import Observation
from rlbench.backend.robot import Robot
from rlbench.backend.waypoints import Waypoint
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import BasketballInHoop

NUM_EPISODES = 5

frames: List[np.ndarray] = []
robot: Optional[Robot] = None
apply_failure: bool = False
failure_steps: int = 10
current_steps: int = 0


def on_waypoint(point: Waypoint) -> None:
    global apply_failure, robot
    print(f"Next waypoint is: {point._waypoint.get_name()}")
    if point._waypoint.get_name() == "waypoint2" and not apply_failure:
        apply_failure = True
        robot = point._robot


def on_step(obs: Observation) -> None:
    global frames, apply_failure, failure_steps, current_steps, robot
    frames.append(obs.front_rgb)
    if apply_failure:
        current_steps += 1
        if current_steps >= failure_steps:
            apply_failure = False
            if robot is not None:
                robot.gripper.release()


def main() -> int:
    global frames

    obs_config = ObservationConfig()
    obs_config.set_all(True)

    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
        ),
        obs_config=ObservationConfig(),
        headless=False,
    )
    env.launch()

    task_env = env.get_task(BasketballInHoop)

    (demo,) = task_env.get_failures(
        amount=1,
        callable_each_waypoint=on_waypoint,
        callable_each_step=on_step,
    )

    if len(frames) > 0:
        rendered_clip = ImageSequenceClip(frames, fps=30)
        rendered_clip.write_videofile("vid_basketball_in_hoop.mp4")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
