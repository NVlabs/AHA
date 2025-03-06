# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import argparse
from multiprocessing import Process
from typing import List

from failgen.env_wrapper import FailGenEnvWrapper
from failgen.fail_grasp import GraspFailure
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
    WrongObjectFailure.FAILURE_TYPE,
]


def run_get_failures(task_name: str, fail_type: str, max_tries: int) -> None:
    env_wrapper = FailGenEnvWrapper(
        task_name=task_name,
        headless=True,
        record=True,
        save_data=False,
    )

    # Set current failure type
    has_failtype = False
    for fail_obj in env_wrapper.manager._failures:
        if fail_obj.failure_type == fail_type:
            fail_obj.set_enabled(True)
            has_failtype = True
        else:
            fail_obj.set_enabled(False)

    if not has_failtype:
        print(f"Skipping task {task_name} and fail {fail_type}")
        env_wrapper.shutdown()
        return

    print(
        f"Starting demo collection for task: {task_name} and fail: {fail_type}"
    )

    attempts = max_tries
    while attempts > 0:
        env_wrapper.reset()
        demo, success = env_wrapper.get_failure()
        if demo is not None and not success:
            env_wrapper.save_video(f"vid_{task_name}_{fail_type}.mp4")
            break
        else:
            attempts -= 1
    if attempts <= 0:
        print(f"Got an issue with task: {task_name}, failure: {fail_type}")
    else:
        print(f"Saved episode for task: {task_name} - failure: {fail_type}")

    env_wrapper.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="basketball_in_hoop",
        help="The name of the task to load for this example",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=10,
        help="The maximum number of tries to test a single failure",
    )
    parser.add_argument(
        "--multiprocessing",
        action="store_true",
        help="Whether or not to use multiprocessing for data collection",
    )

    args = parser.parse_args()

    if args.multiprocessing:
        processes = [
            Process(
                target=run_get_failures,
                args=(
                    args.task,
                    fail_type,
                    args.episodes,
                    args.max_tries,
                    args.video,
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
                max_tries=args.max_tries,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
