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


def run_get_failures(
    task_name: str,
    fail_type: str,
    num_episodes: int,
    max_tries: int,
    save_video: bool,
    save_path: str,
) -> None:
    env_wrapper = FailGenEnvWrapper(
        task_name=task_name,
        headless=True,
        record=save_video,
        save_data=True,
        save_path=save_path,
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

    # for i in range(num_episodes):
    #     env_wrapper.reset()
    #     attempts = max_tries
    #     while attempts > 0:
    #         demo, success = env_wrapper.get_failure()
    #         if demo is not None and not success:
    #             env_wrapper.save_failure_ext(i, fail_type, demo)
    #             env_wrapper.save_video(f"vid_{task_name}_{fail_type}_{i}.mp4")
    #             break
    #         else:
    #             attempts -= 1
    #     if attempts <= 0:
    #       print(f"Got an issue with task: {task_name}, failure: {fail_type}")
    #     else:
    #         print(f"Saved episode {i+1} / {num_episodes}")
    # print(
    #   f"Recorded {num_episodes} for task {task_name} and failure: {fail_type}"
    # )

    # Wrong-object failures are a whole different set from normal failures, so
    # will handle data recording in a separate way
    if fail_type == WrongObjectFailure.FAILURE_TYPE:
        for i in range(num_episodes):
            env_wrapper.reset()
            attempts = max_tries
            while attempts > 0:
                demo, success = env_wrapper.get_failure()
                if demo is not None and not success:
                    env_wrapper.save_cameras(i, fail_type)
                    #### env_wrapper.save_failure_ext(i, fail_type, demo)
                    #### env_wrapper.save_video(
                    ####     f"vid_{task_name}_{fail_type}_{i}.mp4"
                    #### )
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
                    env_wrapper.save_cameras(i, fail_type, wp_idx)
                    #### env_wrapper.save_failure_ext(i, fail_type, demo, wp_idx)
                    #### env_wrapper.save_video(
                    ####     f"vid_{task_name}_{fail_type}_{i}.mp4"
                    #### )
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
        default=10,
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
        "--video",
        action="store_true",
        help="Whether or not to save video recordings of the failures",
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
                    args.video,
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
                save_video=args.video,
                save_path=args.savepath,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
