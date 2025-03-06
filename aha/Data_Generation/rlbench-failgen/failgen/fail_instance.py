# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import abc
from typing import List, Optional

import numpy as np
from pyrep.objects.dummy import Dummy
from rlbench.backend.robot import Robot
from rlbench.backend.task import Task
from rlbench.backend.waypoints import Waypoint


class IFailure(abc.ABC):
    def __init__(
        self,
        robot: Robot,
        name: str,
        waypoints_indices: List[int],
    ):
        self._robot: Robot = robot
        self._name: str = name
        self._obj_base: Optional[Dummy] = None
        self._enabled: bool = True
        self._failure_type = "undefined"

        self._waypoints_indices = waypoints_indices.copy()
        self._waypoints_names = [f"waypoint{idx}" for idx in waypoints_indices]
        self._waypoint_fail_name = np.random.choice(self._waypoints_names)

    def set_obj_base(self, obj_base: Dummy) -> None:
        self._obj_base = obj_base

    def change_waypoint_fail_name(self, waypoint_name) -> None:
        self._waypoint_fail_name = waypoint_name

    @property
    def waypoints_indices(self) -> List[int]:
        return self._waypoints_indices

    @property
    def failure_type(self) -> str:
        return self._failure_type

    def set_enabled(self, value: bool):
        self._enabled = value

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def name(self) -> str:
        return self._name

    @abc.abstractmethod
    def on_start(self, task: Task) -> None:
        """Callable for each start of the environment"""
        ...

    @abc.abstractmethod
    def on_reset(self) -> None:
        """Callable for each reset of the environment"""
        ...

    @abc.abstractmethod
    def on_step(self) -> None:
        """Callable for calling on each simulation step"""
        ...

    @abc.abstractmethod
    def on_waypoint(self, point: Waypoint) -> None:
        """Callable for calling on each change of waypoint"""
        ...
