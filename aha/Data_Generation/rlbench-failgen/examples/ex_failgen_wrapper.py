import argparse

from failgen.env_wrapper import FailGenEnvWrapper


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="basketball_in_hoop",
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
        "--save_data",
        action="store_true",
        help="Whether or not to save the data recorded",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Whether or not to record a separate video",
    )

    args = parser.parse_args()

    env_wrapper = FailGenEnvWrapper(
        task_name=args.task,
        headless=args.headless,
        record=args.record,
        save_data=args.save_data,
    )

    for i in range(args.episodes):
        env_wrapper.reset()
        demo, _ = env_wrapper.get_failure()
        if args.save_data and demo is not None:
            print(f"Saving episode {i+1} / {args.episodes}")
            env_wrapper.save_failure(i, demo)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
