import os
from typing import List, Optional

import numpy as np
from omegaconf import OmegaConf

from failgen.env_wrapper import RLBENCH_TASKPY_FOLDER, FailGenEnvWrapper
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


class WaypointsSweeper:
    def __init__(
        self,
        task_name: str,
        task_folder: str = RLBENCH_TASKPY_FOLDER,
        headless: bool = True,
    ):
        self._task_name: str = task_name
        self._env_wrapper = FailGenEnvWrapper(
            task_name=task_name,
            task_folder=task_folder,
            headless=headless,
            record=True,
            save_data=False,
            no_failures=True,
        )
        self._config = self._env_wrapper.config.copy()
        self._waypoints_indices: List[int] = self._config.data.waypoints

        self._current_failure: Optional[IFailure] = None

    @property
    def waypoint_indices(self) -> List[int]:
        return self._waypoints_indices

    def shutdown(self) -> None:
        self._env_wrapper.shutdown()

    def test_failure(
        self, fail_type: str, max_tries: int, waypoint_idx: int = -1
    ) -> bool:
        got_failure = False
        save_data = None
        fail_name = None
        if fail_type == SlipFailure.FAILURE_TYPE:
            for i in range(max_tries):
                self._env_wrapper.manager.clear()
                self._env_wrapper.reset()

                # Sample random args for SlipFailure constructor
                arg_waypoint_indices = [
                    int(np.random.choice(self._waypoints_indices))
                    if waypoint_idx == -1
                    else waypoint_idx
                ]
                arg_num_steps_till_fail = np.random.randint(low=0, high=10)

                # Make a descriptive name for later logging
                fail_name = "failure_slip_i{}_w{}_n{}".format(
                    i, arg_waypoint_indices[0], arg_num_steps_till_fail
                )

                # Create the failure and add it to the manager manually
                self._current_failure = SlipFailure(
                    robot=self._env_wrapper.robot,
                    name=fail_name,
                    waypoint_indices=arg_waypoint_indices,
                    num_steps_till_fail=arg_num_steps_till_fail,
                )
                self._env_wrapper.manager.add_failure(self._current_failure)

                try:
                    _, success = self._env_wrapper.get_failure()
                except RuntimeError:
                    success = True

                # If failed, then we got the right setting
                if not success:
                    got_failure = True
                    save_data = {
                        "fail_name": fail_name,
                        "waypoint_indices": arg_waypoint_indices,
                        "num_steps_till_fail": arg_num_steps_till_fail,
                    }
                    break

        elif fail_type == GraspFailure.FAILURE_TYPE:
            for i in range(max_tries):
                self._env_wrapper.manager.clear()
                self._env_wrapper.reset()

                # Sample random args for GraspFailure constructor
                arg_waypoint_indices = [
                    int(np.random.choice(self._waypoints_indices))
                    if waypoint_idx == -1
                    else waypoint_idx
                ]

                # Make a descriptive name for later logging
                fail_name = f"failure_grasp_i{i}_w{arg_waypoint_indices[0]}"

                # Create the failure and add it to the manager manually
                self._current_failure = GraspFailure(
                    robot=self._env_wrapper.robot,
                    name=fail_name,
                    waypoint_indices=arg_waypoint_indices,
                )
                self._env_wrapper.manager.add_failure(self._current_failure)

                try:
                    _, success = self._env_wrapper.get_failure()
                except RuntimeError:
                    success = True

                # If failed, then we got the right setting
                if not success:
                    got_failure = True
                    save_data = {
                        "fail_name": fail_name,
                        "waypoint_indices": arg_waypoint_indices,
                    }
                    break

        elif fail_type == TranslationXFailure.FAILURE_TYPE:
            for i in range(max_tries):
                self._env_wrapper.manager.clear()
                self._env_wrapper.reset()

                # Sample random args for GraspFailure constructor
                arg_waypoint_indices = [
                    int(np.random.choice(self._waypoints_indices))
                    if waypoint_idx == -1
                    else waypoint_idx
                ]
                arg_range = (-0.5, 0.5)

                # Make a descriptive name for later logging
                fail_name = "failure_translation_x_i{}_w{}".format(
                    i, arg_waypoint_indices[0]
                )

                # Create the failure and add it to the manager manually
                self._current_failure = TranslationXFailure(
                    robot=self._env_wrapper.robot,
                    name=fail_name,
                    waypoints_indices=arg_waypoint_indices,
                    translation_range=arg_range,
                )
                self._env_wrapper.manager.add_failure(self._current_failure)

                try:
                    _, success = self._env_wrapper.get_failure()
                except RuntimeError:
                    success = True

                # If failed, the we got the right setting
                if not success:
                    got_failure = True
                    save_data = {
                        "fail_name": fail_name,
                        "waypoint_indices": arg_waypoint_indices,
                        "range": arg_range,
                    }
                    break

        elif fail_type == TranslationYFailure.FAILURE_TYPE:
            for i in range(max_tries):
                self._env_wrapper.manager.clear()
                self._env_wrapper.reset()

                # Sample random args for GraspFailure constructor
                arg_waypoint_indices = [
                    int(np.random.choice(self._waypoints_indices))
                    if waypoint_idx == -1
                    else waypoint_idx
                ]
                arg_range = (-0.5, 0.5)

                # Make a descriptive name for later logging
                fail_name = "failure_translation_y_i{}_w{}".format(
                    i, arg_waypoint_indices[0]
                )

                # Create the failure and add it to the manager manually
                self._current_failure = TranslationYFailure(
                    robot=self._env_wrapper.robot,
                    name=fail_name,
                    waypoints_indices=arg_waypoint_indices,
                    translation_range=arg_range,
                )
                self._env_wrapper.manager.add_failure(self._current_failure)

                try:
                    _, success = self._env_wrapper.get_failure()
                except RuntimeError:
                    success = True

                # If failed, the we got the right setting
                if not success:
                    got_failure = True
                    save_data = {
                        "fail_name": fail_name,
                        "waypoint_indices": arg_waypoint_indices,
                        "range": arg_range,
                    }
                    break

        elif fail_type == TranslationZFailure.FAILURE_TYPE:
            for i in range(max_tries):
                self._env_wrapper.manager.clear()
                self._env_wrapper.reset()

                # Sample random args for GraspFailure constructor
                arg_waypoint_indices = [
                    int(np.random.choice(self._waypoints_indices))
                    if waypoint_idx == -1
                    else waypoint_idx
                ]
                arg_range = (-0.5, 0.5)

                # Make a descriptive name for later logging
                fail_name = "failure_translation_z_i{}_w{}".format(
                    i, arg_waypoint_indices[0]
                )

                # Create the failure and add it to the manager manually
                self._current_failure = TranslationZFailure(
                    robot=self._env_wrapper.robot,
                    name=fail_name,
                    waypoints_indices=arg_waypoint_indices,
                    translation_range=arg_range,
                )
                self._env_wrapper.manager.add_failure(self._current_failure)

                try:
                    _, success = self._env_wrapper.get_failure()
                except RuntimeError:
                    success = True

                # If failed, the we got the right setting
                if not success:
                    got_failure = True
                    save_data = {
                        "fail_name": fail_name,
                        "waypoint_indices": arg_waypoint_indices,
                        "range": arg_range,
                    }
                    break

        elif fail_type == RotationXFailure.FAILURE_TYPE:
            for i in range(max_tries):
                self._env_wrapper.manager.clear()
                self._env_wrapper.reset()

                # Sample random args for GraspFailure constructor
                arg_waypoint_indices = [
                    int(np.random.choice(self._waypoints_indices))
                    if waypoint_idx == -1
                    else waypoint_idx
                ]
                arg_range = (-0.5 * np.pi, 0.5 * np.pi)

                # Make a descriptive name for later logging
                fail_name = "failure_rotation_x_i{}_w{}".format(
                    i, arg_waypoint_indices[0]
                )

                # Create the failure and add it to the manager manually
                self._current_failure = RotationXFailure(
                    robot=self._env_wrapper.robot,
                    name=fail_name,
                    waypoints_indices=arg_waypoint_indices,
                    rotation_range=arg_range,
                )
                self._env_wrapper.manager.add_failure(self._current_failure)

                try:
                    _, success = self._env_wrapper.get_failure()
                except RuntimeError:
                    success = True

                # If failed, the we got the right setting
                if not success:
                    got_failure = True
                    save_data = {
                        "fail_name": fail_name,
                        "waypoint_indices": arg_waypoint_indices,
                        "range": arg_range,
                    }
                    break

        elif fail_type == RotationYFailure.FAILURE_TYPE:
            for i in range(max_tries):
                self._env_wrapper.manager.clear()
                self._env_wrapper.reset()

                # Sample random args for GraspFailure constructor
                arg_waypoint_indices = [
                    int(np.random.choice(self._waypoints_indices))
                    if waypoint_idx == -1
                    else waypoint_idx
                ]
                arg_range = (-0.5 * np.pi, 0.5 * np.pi)

                # Make a descriptive name for later logging
                fail_name = "failure_rotation_y_i{}_w{}".format(
                    i, arg_waypoint_indices[0]
                )

                # Create the failure and add it to the manager manually
                self._current_failure = RotationYFailure(
                    robot=self._env_wrapper.robot,
                    name=fail_name,
                    waypoints_indices=arg_waypoint_indices,
                    rotation_range=arg_range,
                )
                self._env_wrapper.manager.add_failure(self._current_failure)

                try:
                    _, success = self._env_wrapper.get_failure()
                except RuntimeError:
                    success = True

                # If failed, the we got the right setting
                if not success:
                    got_failure = True
                    save_data = {
                        "fail_name": fail_name,
                        "waypoint_indices": arg_waypoint_indices,
                        "range": arg_range,
                    }
                    break

        elif fail_type == RotationZFailure.FAILURE_TYPE:
            for i in range(max_tries):
                self._env_wrapper.manager.clear()
                self._env_wrapper.reset()

                # Sample random args for GraspFailure constructor
                arg_waypoint_indices = [
                    int(np.random.choice(self._waypoints_indices))
                    if waypoint_idx == -1
                    else waypoint_idx
                ]
                arg_range = (-0.5 * np.pi, 0.5 * np.pi)

                # Make a descriptive name for later logging
                fail_name = "failure_rotation_z_i{}_w{}".format(
                    i, arg_waypoint_indices[0]
                )

                # Create the failure and add it to the manager manually
                self._current_failure = RotationZFailure(
                    robot=self._env_wrapper.robot,
                    name=fail_name,
                    waypoints_indices=arg_waypoint_indices,
                    rotation_range=arg_range,
                )
                self._env_wrapper.manager.add_failure(self._current_failure)

                try:
                    _, success = self._env_wrapper.get_failure()
                except RuntimeError:
                    success = True

                # If failed, the we got the right setting
                if not success:
                    got_failure = True
                    save_data = {
                        "fail_name": fail_name,
                        "waypoint_indices": arg_waypoint_indices,
                        "range": arg_range,
                    }
                    break

        elif fail_type == NoRotationFailure.FAILURE_TYPE:
            for i in range(max_tries):
                self._env_wrapper.manager.clear()
                self._env_wrapper.reset()

                # Sample random args for GraspFailure constructor
                arg_waypoint_indices = [
                    int(np.random.choice(self._waypoints_indices))
                    if waypoint_idx == -1
                    else waypoint_idx
                ]

                # Make a descriptive name for later logging
                fail_name = "failure_no_rotation_i{}_w{}".format(
                    i, arg_waypoint_indices[0]
                )

                # Create the failure and add it to the manager manually
                self._current_failure = NoRotationFailure(
                    robot=self._env_wrapper.robot,
                    name=fail_name,
                    waypoints_indices=arg_waypoint_indices,
                )
                self._env_wrapper.manager.add_failure(self._current_failure)

                try:
                    _, success = self._env_wrapper.get_failure()
                except RuntimeError:
                    success = True

                # If failed, the we got the right setting
                if not success:
                    got_failure = True
                    save_data = {
                        "fail_name": fail_name,
                        "waypoint_indices": arg_waypoint_indices,
                    }
                    break

        elif fail_type == WrongSequenceFailure.FAILURE_TYPE:
            for i in range(max_tries):
                self._env_wrapper.manager.clear()
                self._env_wrapper.reset()

                # Sample random args for GraspFailure constructor
                valid_waypoints = self._waypoints_indices.copy()
                first_waypoint = int(np.random.choice(valid_waypoints))
                valid_waypoints.remove(first_waypoint)
                second_waypoint = int(np.random.choice(valid_waypoints))
                arg_waypoint_indices = [first_waypoint, second_waypoint]

                # Make a descriptive name for later logging
                fail_name = "failure_wrong_sequence_i{}_w{}_w{}".format(
                    i, first_waypoint, second_waypoint
                )

                # Create the failure and add it to the manager manually
                self._current_failure = WrongSequenceFailure(
                    robot=self._env_wrapper.robot,
                    name=fail_name,
                    waypoints_indices=arg_waypoint_indices,
                )
                self._env_wrapper.manager.add_failure(self._current_failure)

                try:
                    _, success = self._env_wrapper.get_failure()
                except RuntimeError:
                    success = True

                # If failed, the we got the right setting
                if not success:
                    got_failure = True
                    save_data = {
                        "fail_name": fail_name,
                        "waypoint_indices": arg_waypoint_indices,
                    }
                    break

        if save_data is not None and fail_name is not None:
            self._env_wrapper.save_video(
                f"vid_{self._task_name}_{fail_name}.mp4"
            )
            OmegaConf.save(
                OmegaConf.create(save_data),
                os.path.join(
                    self._env_wrapper._savepath,
                    f"{self._task_name}_{fail_type}.yaml",
                ),
            )

        return got_failure
