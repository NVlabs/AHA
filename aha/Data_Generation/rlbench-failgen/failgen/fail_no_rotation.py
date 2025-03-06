# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from enum import Enum
from typing import List

from pyrep.objects.object import Object
from rlbench.backend.robot import Robot
from rlbench.backend.task import Task
from rlbench.backend.waypoints import Waypoint

from failgen.fail_instance import IFailure


class NoRotationState(Enum):
    IDLE = 0
    APPLIED = 1


class NoRotationFailure(IFailure):
    FAILURE_TYPE = "no_rotation"

    def __init__(
        self,
        robot: Robot,
        name: str,
        waypoints_indices: List[int],
    ):
        super().__init__(
            robot=robot, name=name, waypoints_indices=waypoints_indices
        )

        self._failure_type = NoRotationFailure.FAILURE_TYPE
        self._state = NoRotationState.IDLE

    def on_start(self, task: Task) -> None:
        # Do nothing for now
        ...

    def on_reset(self) -> None:
        self._state = NoRotationState.IDLE

    def on_step(self) -> None:
        # Do nothing for now
        ...

    def on_waypoint(self, point: Waypoint) -> None:
        if not self._enabled:
            return
        if self._state == NoRotationState.IDLE:
            if point._waypoint.get_name() == self._waypoint_fail_name:
                self._state = NoRotationState.APPLIED
                # Get the previous waypoint to use its orientation
                curr_waypoint_name = point._waypoint.get_name()
                curr_waypoint_idx = int(curr_waypoint_name[len("waypoint") :])
                prev_waypoint_idx = curr_waypoint_idx - 1
                prev_waypoint_name = f"waypoint{prev_waypoint_idx}"
                prev_waypoint = Object.get_object(prev_waypoint_name)
                point._waypoint.set_orientation(prev_waypoint.get_orientation())
