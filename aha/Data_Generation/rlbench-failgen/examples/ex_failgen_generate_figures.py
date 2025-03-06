# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from argparse import ArgumentParser
from pathlib import Path
import re
import subprocess

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
]

FAILURE_TASKS = [
    "basketball_in_hoop",
    "beat_the_buzz",
    "change_channel",
    "change_clock",
    "close_box",
    "close_door",
    "close_drawer",
    "close_fridge",
    "close_grill",
    "close_jar",
    "close_laptop_lid",
    "close_microwave",
    "get_ice_from_fridge",
    "hang_frame_on_hanger",
    "hit_ball_with_queue",
    "hockey",
    "insert_onto_square_peg",
    "insert_usb_in_computer",
    "lamp_off",
    "lamp_on",
    "lift_numbered_block",
    "light_bulb_in",
    "light_bulb_out",
    "meat_on_grill",
    "move_hanger",
    "open_door",
    "open_drawer",
    "open_jar",
    "open_microwave",
    "open_oven",
    "open_washing_machine",
    "open_window",
    "open_wine_bottle",
    "phone_on_base",
    "pick_and_lift_small",
    "pick_and_lift",
    "pick_up_cup",
    "place_cups",
    "place_hanger_on_rack",
    "place_shape_in_shape_sorter",
    "play_jenga",
    "plug_charger_in_power_supply",
    "pour_from_cup_to_cup",
    "press_switch",
    "push_buttons",
    "push_button",
    "put_bottle_in_fridge",
    "put_groceries_in_cupboard",
    "put_item_in_drawer",
    "put_knife_in_knife_block",
    "put_knife_on_chopping_board",
    "put_money_in_safe",
    "put_toilet_roll_on_stand",
    "reach_and_drag",
    "remove_cups",
    "scoop_with_spatula",
    "screw_nail",
    "setup_checkers",
    "setup_chess",
    "slide_block_to_target",
    "solve_puzzle",
    "stack_blocks",
    "stack_chairs",
    "stack_cups",
    "stack_wine",
    "straighten_rope",
    "sweep_to_dustpan",
    "take_cup_out_from_cabinet",
    "take_frame_off_hanger",
    "take_item_out_of_drawer",
    "take_lid_off_saucepan",
    "take_money_out_safe",
    "take_off_weighing_scales",
    "take_plate_off_colored_dish_rack",
    "take_shoes_out_of_box",
    "take_toilet_roll_off_stand",
    "take_tray_out_of_oven",
    "take_umbrella_out_of_umbrella_stand",
    "take_usb_out_of_computer",
    "toilet_seat_down",
    "toilet_seat_up",
    "turn_oven_on",
    "turn_tap",
    "tv_on",
    "unplug_charger",
    "water_plants",
    "weighing_scales",
    "wipe_desk",
]


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-folder",
        type=str,
        default="/home/gregor/data/rlbench-failgen",
        help="Folder from which to start the figure generation process",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        type=str,
        default="/home/gregor/data/rlbench-failgen-figures",
        help="Folder where to place the generated artifacts of this process",
    )

    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True)

    tasks_folders = [
        path
        for path in input_folder.iterdir()
        if path.is_dir() and path.stem in FAILURE_TASKS
    ]

    failure_pattern = '|'.join(map(re.escape, FAILURE_TYPES))
    pattern = rf'^vid_(.+?)_({failure_pattern})\.mp4$'
    regex = re.compile(pattern)
    valid_files = []
    for task_folder in tasks_folders:
        task_name = task_folder.stem
        files = [file for file in task_folder.iterdir() if file.is_file()]
        for file in files:
            match = regex.match(file.name)
            if match:
                failure_type = match.group(2)
                valid_files.append((task_name, failure_type, file))
            else:
                print(f"No match found for filename: {file.name}")

    for (file_task, fail_type, file_path) in valid_files:
        # Create a folder to store each image per failure type
        save_folder = output_folder / fail_type
        save_folder.mkdir(exist_ok=True)
        image_name = f"img_{file_task}_{fail_type}.jpg"
        output_image = (save_folder / image_name)
        # Decode the last frame using ffmpeg
        if subprocess.call(
            (
                "ffmpeg",
                "-sseof",
                "-0.1",
                "-i",
                file_path.resolve(),
                "-update",
                "1",
                "-q:v",
                "2",
                output_image.resolve()
            )
        ):
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
