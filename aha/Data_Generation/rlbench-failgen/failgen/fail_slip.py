from enum import Enum
from typing import List

from rlbench.backend.robot import Robot
from rlbench.backend.task import Task
from rlbench.backend.waypoints import Waypoint

from failgen.fail_instance import IFailure


class SlipState(Enum):
    IDLE = 0
    PRE_FAIL = 1
    POST_FAIL = 2


class SlipFailure(IFailure):
    FAILURE_TYPE = "slip"

    def __init__(
        self,
        robot: Robot,
        name: str,
        waypoint_indices: List[int],
        num_steps_till_fail: int = 1,
    ):
        super().__init__(
            robot=robot, name=name, waypoints_indices=waypoint_indices
        )

        self._failure_type = SlipFailure.FAILURE_TYPE
        self._state = SlipState.IDLE
        self._steps_counter = 0
        self._max_steps_till_failure = num_steps_till_fail

    def on_start(self, task: Task) -> None:
        # Do nothing for now
        ...

    def on_reset(self) -> None:
        self._state = SlipState.IDLE
        self._steps_counter = 0

    def on_step(self) -> None:
        if not self._enabled:
            return
        if self._state == SlipState.PRE_FAIL:
            self._steps_counter += 1
            if self._steps_counter >= self._max_steps_till_failure:
                self._steps_counter = 0
                self._state = SlipState.POST_FAIL
                self._robot.gripper.release()
                self._robot.gripper.actuate(amount=1.0, velocity=0.1)

    def on_waypoint(self, point: Waypoint) -> None:
        if not self._enabled:
            return
        if self._state == SlipState.IDLE:
            if point._waypoint.get_name() == self._waypoint_fail_name:
                self._state = SlipState.PRE_FAIL
