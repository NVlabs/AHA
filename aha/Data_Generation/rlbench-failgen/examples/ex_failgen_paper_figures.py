import argparse

from failgen.env_wrapper import FailGenEnvWrapper


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="insert_onto_square_peg",
        help="The name of the task to load for this example",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Whether or not to run in headless mode",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="The number of episodes to try",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Whether or not to record a separate video",
    )
    parser.add_argument(
        "--failures",
        action="store_true",
        help="Whether or not to collect failures",
    )

    args = parser.parse_args()

    env_wrapper = FailGenEnvWrapper(
        task_name=args.task,
        headless=args.headless,
        record=args.record,
        save_data=False,
    )

    for i in range(args.episodes):
        env_wrapper.reset()
        if args.failures:
            _, _ = env_wrapper.get_failure()
        else:
            _ = env_wrapper.get_success()

        if args.record:
            tag = "failure" if args.failures else "success"
            env_wrapper.save_video(f"vid_{args.task}_ep_{i+1}_{tag}.mp4")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
