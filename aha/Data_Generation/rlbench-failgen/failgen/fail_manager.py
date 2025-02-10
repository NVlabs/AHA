from typing import List, Optional

import numpy as np
from omegaconf import DictConfig, ListConfig
from pyrep.objects.dummy import Dummy
from rlbench.backend.robot import Robot
from rlbench.backend.task import Task
from rlbench.backend.waypoints import Waypoint

from failgen.fail_grasp import GraspFailure
from failgen.fail_instance import IFailure
from failgen.fail_no_rotation import NoRotationFailure
from failgen.fail_rotation import (
    RotationFailure,
    RotationXFailure,
    RotationYFailure,
    RotationZFailure,
)
from failgen.fail_sequence import WrongSequenceFailure
from failgen.fail_slip import SlipFailure
from failgen.fail_translation import (
    TranslationFailure,
    TranslationXFailure,
    TranslationYFailure,
    TranslationZFailure,
)
from failgen.fail_wrong_object import WrongObjectFailure


class Manager:
    def __init__(
        self,
        robot: Robot,
        obj_base: Dummy,
        fails_cfg: ListConfig = ListConfig([]),
    ):
        self._robot = robot
        self._obj_base = obj_base
        self._failures: List[IFailure] = []

        for fail_cfg in fails_cfg:
            self.create(
                fail_type=fail_cfg.get("type", ""),
                fail_name=fail_cfg.get("name", ""),
                fail_enabled=fail_cfg.get("enabled", False),
                fail_cfg=fail_cfg,
            )

    def add_failure(self, failure: IFailure) -> None:
        self._failures.append(failure)

    def create(
        self,
        fail_type: str,
        fail_name: str,
        fail_enabled: bool,
        fail_cfg: DictConfig,
    ) -> Optional[IFailure]:
        failure: Optional[IFailure] = None
        if fail_type == SlipFailure.FAILURE_TYPE:
            failure = SlipFailure(
                robot=self._robot,
                name=fail_name,
                waypoint_indices=fail_cfg.get("waypoints", []),
                num_steps_till_fail=fail_cfg.get("fail_after", 1),
            )
        elif fail_type == GraspFailure.FAILURE_TYPE:
            failure = GraspFailure(
                robot=self._robot,
                name=fail_name,
                waypoint_indices=fail_cfg.get("waypoints", []),
            )
        elif fail_type == TranslationFailure.FAILURE_TYPE:
            failure = TranslationFailure(
                robot=self._robot,
                name=fail_name,
                waypoints_indices=fail_cfg.get("waypoints", []),
                translation_axis=fail_cfg.get("axis", "x"),
                translation_range=fail_cfg.get("range", (-0.1, 0.1)),
            )
        elif fail_type == TranslationXFailure.FAILURE_TYPE:
            failure = TranslationXFailure(
                robot=self._robot,
                name=fail_name,
                waypoints_indices=fail_cfg.get("waypoints", []),
                translation_range=fail_cfg.get("range", (-0.1, 0.1)),
            )
        elif fail_type == TranslationYFailure.FAILURE_TYPE:
            failure = TranslationYFailure(
                robot=self._robot,
                name=fail_name,
                waypoints_indices=fail_cfg.get("waypoints", []),
                translation_range=fail_cfg.get("range", (-0.1, 0.1)),
            )
        elif fail_type == TranslationZFailure.FAILURE_TYPE:
            failure = TranslationZFailure(
                robot=self._robot,
                name=fail_name,
                waypoints_indices=fail_cfg.get("waypoints", []),
                translation_range=fail_cfg.get("range", (-0.1, 0.1)),
            )
        elif fail_type == RotationFailure.FAILURE_TYPE:
            failure = RotationFailure(
                robot=self._robot,
                name=fail_name,
                waypoints_indices=fail_cfg.get("waypoints", []),
                rotation_axis=fail_cfg.get("axis", "x"),
                rotation_range=fail_cfg.get(
                    "range", (-0.1 * np.pi, 0.1 * np.pi)
                ),
            )
        elif fail_type == RotationXFailure.FAILURE_TYPE:
            failure = RotationXFailure(
                robot=self._robot,
                name=fail_name,
                waypoints_indices=fail_cfg.get("waypoints", []),
                rotation_range=fail_cfg.get(
                    "range", (-0.1 * np.pi, 0.1 * np.pi)
                ),
            )
        elif fail_type == RotationYFailure.FAILURE_TYPE:
            failure = RotationYFailure(
                robot=self._robot,
                name=fail_name,
                waypoints_indices=fail_cfg.get("waypoints", []),
                rotation_range=fail_cfg.get(
                    "range", (-0.1 * np.pi, 0.1 * np.pi)
                ),
            )
        elif fail_type == RotationZFailure.FAILURE_TYPE:
            failure = RotationZFailure(
                robot=self._robot,
                name=fail_name,
                waypoints_indices=fail_cfg.get("waypoints", []),
                rotation_range=fail_cfg.get(
                    "range", (-0.1 * np.pi, 0.1 * np.pi)
                ),
            )
        elif fail_type == WrongObjectFailure.FAILURE_TYPE:
            failure = WrongObjectFailure(
                robot=self._robot,
                name=fail_name,
                waypoints_indices=fail_cfg.get("waypoints", []),
                original_name=fail_cfg.get("original_name", ""),
                alternatives_names=fail_cfg.get("alternatives_names", []),
                key_waypoint=fail_cfg.get("key_waypoint"),
            )
        elif fail_type == NoRotationFailure.FAILURE_TYPE:
            failure = NoRotationFailure(
                robot=self._robot,
                name=fail_name,
                waypoints_indices=fail_cfg.get("waypoints", []),
            )
        elif fail_type == WrongSequenceFailure.FAILURE_TYPE:
            failure = WrongSequenceFailure(
                robot=self._robot,
                name=fail_name,
                waypoints_indices=fail_cfg.get("waypoints", []),
            )

        if failure is not None:
            failure.set_enabled(fail_enabled)
            failure.set_obj_base(self._obj_base)
            self._failures.append(failure)

        return failure

    def clear(self) -> None:
        self._failures.clear()

    def on_start(self, task: Task) -> None:
        for failure in self._failures:
            failure.on_start(task)

    def on_reset(self) -> None:
        for failure in self._failures:
            failure.on_reset()

    def on_step(self) -> None:
        for failure in self._failures:
            failure.on_step()

    def on_waypoint(self, point: Waypoint) -> None:
        for failure in self._failures:
            failure.on_waypoint(point)
