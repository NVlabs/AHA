from enum import Enum
from typing import List

from rlbench.backend.robot import Robot
from rlbench.backend.task import Task
from rlbench.backend.waypoints import Waypoint

from failgen.fail_instance import IFailure


class GraspState(Enum):
    IDLE = 0
    FAIL = 1


class GraspFailure(IFailure):
    FAILURE_TYPE = "grasp"

    def __init__(
        self,
        robot: Robot,
        name: str,
        waypoint_indices: List[int],
    ):
        super().__init__(
            robot=robot, name=name, waypoints_indices=waypoint_indices
        )

        self._failure_type = GraspFailure.FAILURE_TYPE
        self._state = GraspState.IDLE

    def on_start(self, task: Task) -> None:
        # Do nothing for now
        ...

    def on_reset(self) -> None:
        self._state = GraspState.IDLE

    def on_step(self) -> None:
        # Do nothing for now
        ...

    def on_waypoint(self, point: Waypoint) -> None:
        if not self._enabled:
            return
        if self._state == GraspState.IDLE:
            if point._waypoint.get_name() == self._waypoint_fail_name:
                self._state = GraspState.FAIL
                # Force the extension string of the waypoint to be empty, which
                # effectively deactivate open and close gripper commands
                point.clear_ext()
