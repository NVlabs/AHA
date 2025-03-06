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


class TranslationState(Enum):
    IDLE = 0
    APPLIED = 1


class TranslationFailure(IFailure):
    FAILURE_TYPE = "translation"

    def __init__(
        self,
        robot: Robot,
        name: str,
        waypoints_indices: List[int],
        translation_axis: str = "x",
        translation_range: Tuple[float, float] = (-0.1, 0.1),
    ):
        super().__init__(
            robot=robot, name=name, waypoints_indices=waypoints_indices
        )

        self._failure_type = TranslationFailure.FAILURE_TYPE
        self._axis = translation_axis
        self._range = translation_range
        self._state = TranslationState.IDLE

    def on_start(self, task: Task) -> None:
        # Do nothing for now
        ...

    def on_reset(self) -> None:
        self._state = TranslationState.IDLE

    def on_step(self) -> None:
        # Do nothing for now
        ...

    def on_waypoint(self, point: Waypoint) -> None:
        if not self._enabled:
            return
        if self._state == TranslationState.IDLE:
            if point._waypoint.get_name() == self._waypoint_fail_name:
                self._state = TranslationState.APPLIED
                # Apply the translation perturbation in the corresponding axis
                delta = np.random.uniform(self._range[0], self._range[1])
                position = point._waypoint.get_position()
                if self._axis == "x":
                    position[0] += delta
                elif self._axis == "y":
                    position[1] += delta
                elif self._axis == "z":
                    position[2] += delta
                point._waypoint.set_position(position)


class TranslationXFailure(TranslationFailure):
    FAILURE_TYPE = "translation_x"

    def __init__(
        self,
        robot: Robot,
        name: str,
        waypoints_indices: List[int],
        translation_range: Tuple[float, float] = (-0.1, 0.1),
    ):
        super().__init__(
            robot=robot,
            name=name,
            waypoints_indices=waypoints_indices,
            translation_axis="x",
            translation_range=translation_range,
        )

        self._failure_type = TranslationXFailure.FAILURE_TYPE


class TranslationYFailure(TranslationFailure):
    FAILURE_TYPE = "translation_y"

    def __init__(
        self,
        robot: Robot,
        name: str,
        waypoints_indices: List[int],
        translation_range: Tuple[float, float] = (-0.1, 0.1),
    ):
        super().__init__(
            robot=robot,
            name=name,
            waypoints_indices=waypoints_indices,
            translation_axis="y",
            translation_range=translation_range,
        )

        self._failure_type = TranslationYFailure.FAILURE_TYPE


class TranslationZFailure(TranslationFailure):
    FAILURE_TYPE = "translation_z"

    def __init__(
        self,
        robot: Robot,
        name: str,
        waypoints_indices: List[int],
        translation_range: Tuple[float, float] = (-0.1, 0.1),
    ):
        super().__init__(
            robot=robot,
            name=name,
            waypoints_indices=waypoints_indices,
            translation_axis="z",
            translation_range=translation_range,
        )

        self._failure_type = TranslationZFailure.FAILURE_TYPE
