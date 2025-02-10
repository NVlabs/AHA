import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, List

FAILURE_TYPES = [
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
    "wrong_object",
]


def main() -> int:
    current_dir = Path(__file__).parent.resolve()
    filepath_json = (current_dir.parent / "out_of_domain.json").resolve()

    with open(filepath_json, "r") as fhandle:
        data = json.load(fhandle)

    # Grab all image paths from the json data
    images_paths: List[str] = [obj["image"] for obj in data]
    images_names = [
        (img_path.split("/")[-1]).split(".")[0] for img_path in images_paths
    ]

    # Get the counts for each task ocurrance
    tasks_hist = defaultdict(lambda: 0)
    fails_hist = defaultdict(lambda: 0)
    failure_pattern = '|'.join(map(re.escape, FAILURE_TYPES))
    pattern = rf'^([\w_]+?)_({failure_pattern})(?:_wp\d+_|_|$)'
    for image_name in images_names:
        match = re.match(pattern, image_name)
        if match:
            task_name = match.group(1)
            failure_type = match.group(2)

            tasks_hist[task_name] += 1
            fails_hist[failure_type] += 1

    with open(current_dir / "distributions.json", "w") as fhandle:
        data_dist = {
            "task_histogram": tasks_hist,
            "failures_histogram": fails_hist
        }
        json.dump(data_dist, fhandle, indent=4)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
