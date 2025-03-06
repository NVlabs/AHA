# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import argparse
from multiprocessing import Process
from typing import List

from failgen.waypoint_sweeper import WaypointsSweeper

FAILURES_LIST: List[str] = [
    "grasp",
    "slip",
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "translation_x",
    "translation_y",
    "translation_z",
    "no_rotation",
    "wrong_sequence",
]


def run(fail_type: str, task_name: str, headless: bool, max_tries: int) -> None:
    sweeper = WaypointsSweeper(task_name=task_name, headless=headless)
    if fail_type == "wrong_sequence":
        sweeper.test_failure(fail_type, max_tries)
    else:
        for waypoint_idx in sweeper.waypoint_indices:
            sweeper.test_failure(fail_type, max_tries, waypoint_idx)
    sweeper.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="basketball_in_hoop",
        help="The name of the task to load for the sweeper",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Whether or not to run in headless mode",
    )
    parser.add_argument(
        "--max_tries",
        type=int,
        default=2,
        help="The maximum number of tries to test a single failure",
    )

    args = parser.parse_args()

    # sweeper = WaypointsSweeper(task_name=args.task, headless=args.headless)
    # for fail_type in FAILURES_LIST:
    #     if fail_type == "wrong_sequence":
    #         sweeper.test_failure("wrong_sequence", args.max_tries)
    #     else:
    #         for wp_idx in sweeper.waypoint_indices:
    #             sweeper.test_failure(fail_type, args.max_tries, wp_idx)

    processes = [
        Process(
            target=run,
            args=(fail_type, args.task, args.headless, args.max_tries),
        )
        for fail_type in FAILURES_LIST
    ]
    [t.start() for t in processes]
    [t.join() for t in processes]

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
