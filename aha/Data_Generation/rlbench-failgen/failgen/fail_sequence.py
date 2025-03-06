# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from typing import List

from rlbench.backend.robot import Robot
from rlbench.backend.task import Task
from rlbench.backend.waypoints import Waypoint

from failgen.fail_instance import IFailure


class WrongSequenceFailure(IFailure):
    FAILURE_TYPE = "wrong_sequence"

    def __init__(
        self,
        robot: Robot,
        name: str,
        waypoints_indices: List[int],
    ):
        super().__init__(
            robot=robot, name=name, waypoints_indices=waypoints_indices
        )

        self._failure_type = WrongSequenceFailure.FAILURE_TYPE
        self._indices_to_swap = waypoints_indices.copy()
        assert (
            len(self._indices_to_swap) == 2
        ), "WrongSequenceFailure >> must give two points to swap"

    def on_start(self, task: Task) -> None:
        if not self._enabled:
            return

        assert (
            task._waypoints is not None
        ), "WrongSequenceFailure::on_start >> Must have waypoints loaded"

        idx_from, idx_to = self._indices_to_swap

        tmp = task._waypoints[idx_from]
        task._waypoints[idx_from] = task._waypoints[idx_to]
        task._waypoints[idx_to] = tmp

    def on_reset(self) -> None:
        # Do nothing for now
        ...

    def on_step(self) -> None:
        # Do nothing for now
        ...

    def on_waypoint(self, point: Waypoint) -> None:
        # Do nothing for now
        ...
