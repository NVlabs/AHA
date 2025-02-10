#!/usr/bin/env bash

if [ $# -eq 0 ]
  then
    echo "Collecting videos for all tasks"


    tasks=(
        "basketball_in_hoop"
        "beat_the_buzz"
        "change_channel"
        "change_clock"
        "close_box"
        "close_door"
        "close_drawer"
        "close_fridge"
        "close_grill"
        "close_jar"
        "close_laptop_lid"
        "close_microwave"
        "get_ice_from_fridge"
        "hang_frame_on_hanger"
        "hit_ball_with_queue"
        "hockey"
        "insert_onto_square_peg"
        "insert_usb_in_computer"
        "lamp_off"
        "lamp_on"
        "lift_numbered_block"
        "light_bulb_in"
        "light_bulb_out"
        "meat_on_grill"
        "move_hanger"
        "open_door"
        "open_drawer"
        "open_jar"
        "open_microwave"
        "open_oven"
        "open_washing_machine"
        "open_window"
        "open_wine_bottle"
        "phone_on_base"
        "pick_and_lift_small"
        "pick_and_lift"
        "pick_up_cup"
        "place_cups"
        "place_hanger_on_rack"
        "place_shape_in_shape_sorter"
        "play_jenga"
        "plug_charger_in_power_supply"
        "pour_from_cup_to_cup"
        "press_switch"
        "push_buttons"
        "push_button"
        "put_bottle_in_fridge"
        "put_groceries_in_cupboard"
        "put_item_in_drawer"
        "put_knife_in_knife_block"
        "put_knife_on_chopping_board"
        "put_money_in_safe"
        "put_toilet_roll_on_stand"
        "reach_and_drag"
        "remove_cups"
        "scoop_with_spatula"
        "screw_nail"
        "setup_checkers"
        "setup_chess"
        "slide_block_to_target"
        "solve_puzzle"
        "stack_blocks"
        "stack_chairs"
        "stack_cups"
        "stack_wine"
        "straighten_rope"
        "sweep_to_dustpan"
        "take_cup_out_from_cabinet"
        "take_frame_off_hanger"
        "take_item_out_of_drawer"
        "take_lid_off_saucepan"
        "take_money_out_safe"
        "take_off_weighing_scales"
        "take_plate_off_colored_dish_rack"
        "take_shoes_out_of_box"
        "take_toilet_roll_off_stand"
        "take_tray_out_of_oven"
        "take_umbrella_out_of_umbrella_stand"
        "take_usb_out_of_computer"
        "toilet_seat_down"
        "toilet_seat_up"
        "turn_oven_on"
        "turn_tap"
        "tv_on"
        "unplug_charger"
        "water_plants"
        "weighing_scales"
        "wipe_desk"
    )

else
    echo "Collecting videos from task $1"
    tasks=("$1")
fi

SAVE_PATH="/tmp/data/rlbench_data"
NUMBER_OF_EPISODES=100

for task in "${tasks[@]}"
do
    python examples/ex_custom_dataset_video_generator.py \
        --task $task \
        --savepath "$SAVE_PATH" \
        --episodes $NUMBER_OF_EPISODES
done
