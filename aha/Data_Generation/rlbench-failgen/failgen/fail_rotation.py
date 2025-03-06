# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from enum import Enum
from typing import List, Tuple

import numpy as np
from rlbench.backend.robot import Robot
from rlbench.backend.task import Task
from rlbench.backend.waypoints import Waypoint

from failgen.fail_instance import IFailure


class RotationState(Enum):
    IDLE = 0
    APPLIED = 1


class RotationFailure(IFailure):
    FAILURE_TYPE = "rotation"

    def __init__(
        self,
        robot: Robot,
        name: str,
        waypoints_indices: List[int],
        rotation_axis: str = "x",
        rotation_range: Tuple[float, float] = (-0.1 * np.pi, 0.1 * np.pi),
    ):
        super().__init__(
            robot=robot, name=name, waypoints_indices=waypoints_indices
        )

        self._failure_type = RotationFailure.FAILURE_TYPE
        self._axis = rotation_axis
        self._range = rotation_range
        self._state = RotationState.IDLE

    def on_start(self, task: Task) -> None:
        # Do nothing for now
        ...

    def on_reset(self) -> None:
        self._state = RotationState.IDLE

    def on_step(self) -> None:
        # Do nothing for now
        ...

    def on_waypoint(self, point: Waypoint) -> None:
        if not self._enabled:
            return
        if self._state == RotationState.IDLE:
            if point._waypoint.get_name() == self._waypoint_fail_name:
                self._state = RotationState.APPLIED
                # Apply the translation perturbation in the corresponding axis
                delta = np.random.uniform(self._range[0], self._range[1])
                orientation = point._waypoint.get_orientation()
                if self._axis == "x":
                    orientation[0] += delta
                elif self._axis == "y":
                    orientation[1] += delta
                elif self._axis == "z":
                    orientation[2] += delta
                point._waypoint.set_orientation(orientation)


class RotationXFailure(RotationFailure):
    FAILURE_TYPE = "rotation_x"

    def __init__(
        self,
        robot: Robot,
        name: str,
        waypoints_indices: List[int],
        rotation_range: Tuple[float, float] = (-0.1 * np.pi, 0.1 * np.pi),
    ):
        super().__init__(
            robot=robot,
            name=name,
            waypoints_indices=waypoints_indices,
            rotation_axis="x",
            rotation_range=rotation_range,
        )

        self._failure_type = RotationXFailure.FAILURE_TYPE


class RotationYFailure(RotationFailure):
    FAILURE_TYPE = "rotation_y"

    def __init__(
        self,
        robot: Robot,
        name: str,
        waypoints_indices: List[int],
        rotation_range: Tuple[float, float] = (-0.1 * np.pi, 0.1 * np.pi),
    ):
        super().__init__(
            robot=robot,
            name=name,
            waypoints_indices=waypoints_indices,
            rotation_axis="y",
            rotation_range=rotation_range,
        )

        self._failure_type = RotationYFailure.FAILURE_TYPE


class RotationZFailure(RotationFailure):
    FAILURE_TYPE = "rotation_z"

    def __init__(
        self,
        robot: Robot,
        name: str,
        waypoints_indices: List[int],
        rotation_range: Tuple[float, float] = (-0.1 * np.pi, 0.1 * np.pi),
    ):
        super().__init__(
            robot=robot,
            name=name,
            waypoints_indices=waypoints_indices,
            rotation_axis="z",
            rotation_range=rotation_range,
        )

        self._failure_type = RotationZFailure.FAILURE_TYPE
