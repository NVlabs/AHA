# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].


"""
Module to collect failure demonstration data for simulation tasks.

This module is designed to work with the FailGen framework to generate failure
demonstrations from various failure types. It provides functionality to collect
episodes of failure data, save keyframe data, and manage different failure modes,
with support for multiprocessing if enabled.

The primary functions include:
    - run_get_failures: Executes data collection for a specified task and failure type.
    - main: Parses command-line arguments and initiates the data collection process.

Usage:
    Run the script from the command line as follows:
        python script_name.py --task <task_name> --episodes <num_episodes> --max_tries <max_attempts>
            [--multiprocessing] [--failtype <failure_type>] [--savepath <save_directory>]
"""
import argparse
from multiprocessing import Process
from typing import List, Optional

from failgen.env_wrapper import FailGenEnvWrapper
from failgen.fail_grasp import GraspFailure
from failgen.fail_instance import IFailure
from failgen.fail_no_rotation import NoRotationFailure
from failgen.fail_rotation import (
    RotationXFailure,
    RotationYFailure,
    RotationZFailure,
)
from failgen.fail_wrong_object import WrongObjectFailure
from failgen.fail_sequence import WrongSequenceFailure
from failgen.fail_slip import SlipFailure
from failgen.fail_translation import (
    TranslationXFailure,
    TranslationYFailure,
    TranslationZFailure,
)

FAILURES_LIST: List[str] = [
    GraspFailure.FAILURE_TYPE,
    SlipFailure.FAILURE_TYPE,
    RotationXFailure.FAILURE_TYPE,
    RotationYFailure.FAILURE_TYPE,
    RotationZFailure.FAILURE_TYPE,
    TranslationXFailure.FAILURE_TYPE,
    TranslationYFailure.FAILURE_TYPE,
    TranslationZFailure.FAILURE_TYPE,
    NoRotationFailure.FAILURE_TYPE,
    WrongSequenceFailure.FAILURE_TYPE,
    WrongObjectFailure.FAILURE_TYPE,
]
"""
    Collect failure demonstration data for a given task and failure type.

    This function sets up a simulation environment for the specified task and then
    enables only the provided failure type while disabling all others. It attempts to
    collect a specified number of failure demonstration episodes. For each episode,
    the environment is reset, and a failure demonstration is captured within a given
    number of tries. Special handling is implemented for the 'WrongObjectFailure' type,
    which uses a separate data recording approach.

    Parameters:
        task_name (str): Name of the task for which to collect the demonstration.
        fail_type (str): The failure type to trigger and collect data for.
        num_episodes (int): The number of episodes (data samples) to collect.
        max_tries (int): The maximum number of attempts per episode to trigger the failure.
        save_path (str): The directory path where the collected keyframe data will be saved.

    Returns:
        None
    """

def run_get_failures(
    task_name: str,
    fail_type: str,
    num_episodes: int,
    max_tries: int,
    save_path: str,
) -> None:
    env_wrapper = FailGenEnvWrapper(
        task_name=task_name,
        headless=True,
        record=False,
        save_data=True,
        save_path=save_path,
        save_keyframes_only=True,
    )

    # Set current failure type
    has_failtype = False
    target_fail_obj: Optional[IFailure] = None
    for fail_obj in env_wrapper.manager._failures:
        if fail_obj.failure_type == fail_type:
            fail_obj.set_enabled(True)
            has_failtype = True
            target_fail_obj = fail_obj
        else:
            fail_obj.set_enabled(False)

    if not has_failtype:
        print(f"Skipping task {task_name} and fail {fail_type}")
        env_wrapper.shutdown()
        return

    print(
        f"Starting demo collection for task: {task_name} and fail: {fail_type}"
    )

    # Wrong-object failures are a whole different set from normal failures, so
    # will handle data recording in a separate way
    if fail_type == WrongObjectFailure.FAILURE_TYPE:
        for i in range(num_episodes):
            env_wrapper.reset()
            attempts = max_tries
            while attempts > 0:
                demo, success = env_wrapper.get_failure()
                if demo is not None and not success:
                    env_wrapper.save_keyframe_data(i, fail_type)
                    break
                else:
                    attempts -= 1
            if attempts <= 0:
                print(
                    f"Got an issue with task: {task_name}, failure: {fail_type}"
                )
            else:
                print(f"Saved episode {i+1} / {num_episodes}")
        print(
            f"Saved {num_episodes} for task {task_name}, failure: {fail_type}"
        )
        return

    assert target_fail_obj is not None
    potential_waypoints = target_fail_obj.waypoints_indices

    for wp_idx in potential_waypoints:
        target_fail_obj.change_waypoint_fail_name(f"waypoint{wp_idx}")
        print(f"Triying to collect from waypoint {wp_idx}")
        for i in range(num_episodes):
            env_wrapper.reset()
            attempts = max_tries
            while attempts > 0:
                demo, success = env_wrapper.get_failure()
                if demo is not None and not success:
                    env_wrapper.save_keyframe_data(i, fail_type, wp_idx)
                    break
                else:
                    attempts -= 1
            if attempts <= 0:
                print(
                    f"Got an issue with task: {task_name}, failure: {fail_type}"
                )
            else:
                print(f"Saved episode {i+1} / {num_episodes}")
        print(
            f"Saved {num_episodes} for task {task_name}, failure: {fail_type}, "
            + f"waypoint-index: {wp_idx}"
        )

    env_wrapper.shutdown()
"""
Entry point for the failure data collection script.

This function parses command-line arguments to determine the parameters for data
collection, including task name, number of episodes, maximum tries per episode,
whether to use multiprocessing, failure type, and the save path. Based on these
arguments, it then initiates the data collection process either in a sequential
or multiprocessing mode for each failure type specified.

Returns:
    int: An exit status code (0 indicates successful execution).
"""

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="basketball_in_hoop",
        help="The name of the task to load for this example",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="The number of episodes to collect",
    )
    parser.add_argument(
        "--max_tries",
        type=int,
        default=10,
        help="The maximum number of tries to test a single failure",
    )
    parser.add_argument(
        "--multiprocessing",
        action="store_true",
        help="Whether or not to use multiprocessing for data collection",
    )
    parser.add_argument(
        "--failtype",
        type=str,
        default="",
        help="The fail type to use for data collection of single failure"
    )
    parser.add_argument(
        "--savepath",
        type=str,
        default="",
        help="The path to the folder where to save all the data",
    )

    args = parser.parse_args()

    global FAILURES_LIST
    if args.failtype != "":
        FAILURES_LIST = [args.failtype]

    if args.multiprocessing:
        processes = [
            Process(
                target=run_get_failures,
                args=(
                    args.task,
                    fail_type,
                    args.episodes,
                    args.max_tries,
                    args.savepath,
                ),
            )
            for fail_type in FAILURES_LIST
        ]
        [t.start() for t in processes]
        [t.join() for t in processes]
    else:
        for fail_type in FAILURES_LIST:
            run_get_failures(
                task_name=args.task,
                fail_type=fail_type,
                num_episodes=args.episodes,
                max_tries=args.max_tries,
                save_path=args.savepath,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
