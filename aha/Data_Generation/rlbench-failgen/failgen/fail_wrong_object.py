# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from enum import Enum
from typing import List

import numpy as np
from pyrep.objects.object import Object
from rlbench.backend.robot import Robot
from rlbench.backend.task import Task
from rlbench.backend.waypoints import Waypoint

from failgen.fail_instance import IFailure


class WrongObjectState(Enum):
    IDLE = 0
    APPLIED = 1


class WrongObjectFailure(IFailure):
    FAILURE_TYPE = "wrong_object"

    def __init__(
        self,
        robot: Robot,
        name: str,
        waypoints_indices: List[int],
        original_name: str,
        alternatives_names: List[str],
        key_waypoint: str,
    ):
        super().__init__(
            robot=robot, name=name, waypoints_indices=waypoints_indices
        )

        self._failure_type = WrongObjectFailure.FAILURE_TYPE
        self._state = WrongObjectState.IDLE
        self._original_name: str = original_name
        self._alternatives_names: List[str] = alternatives_names
        self._key_waypoint: str = key_waypoint

        self._waypoints_original_parents = {}
        self._waypoints_original_pose = {}

        self._num_fix_steps = 10

    def on_start(self, task: Task) -> None:
        if not self._enabled:
            return

        if self._state == WrongObjectState.IDLE:
            self._state = WrongObjectState.APPLIED
            original_obj = Object.get_object(self._original_name)
            alternative_obj = Object.get_object(
                np.random.choice(self._alternatives_names)
            )

            waypoints_names = [
                f"waypoint{idx}" for idx in self._waypoints_indices
            ]
            waypoints_objs = {
                wp_name: Object.get_object(wp_name)
                for wp_name in waypoints_names
            }
            waypoints_rel_poses = {
                wp_name: wp_obj.get_pose(relative_to=original_obj)
                for (wp_name, wp_obj) in waypoints_objs.items()
            }

            for wp_name in waypoints_objs:
                self._waypoints_original_parents[wp_name] = waypoints_objs[
                    wp_name
                ].get_parent()
                self._waypoints_original_pose[wp_name] = waypoints_objs[
                    wp_name
                ].get_pose()
                waypoints_objs[wp_name].set_parent(self._obj_base)
                waypoints_objs[wp_name].set_pose(
                    waypoints_rel_poses[wp_name], relative_to=alternative_obj
                )

    def on_reset(self) -> None:
        self._state = WrongObjectState.IDLE

        for wp_name in self._waypoints_original_parents:
            waypoint_obj = Object.get_object(wp_name)
            waypoint_obj.set_pose(self._waypoints_original_pose[wp_name])
            waypoint_obj.set_parent(self._waypoints_original_parents[wp_name])

        self._num_fix_steps = 10
        self._waypoints_original_parents = {}
        self._waypoints_original_pose = {}

    def on_step(self) -> None:
        # Do nothing for now
        ...

    def on_waypoint(self, point: Waypoint) -> None:
        # Do nothing for now
        ...
