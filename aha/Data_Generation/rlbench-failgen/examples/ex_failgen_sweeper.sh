#!/usr/bin/env bash

if [ $# -eq 0 ]
  then
    echo "Checking settings for all tasks"

    # tasks=("basketball_in_hoop"
    #        "close_box"
    #        "close_jar"
    #        "close_laptop_lid"
    #        "get_ice_from_fridge"
    #        "hockey"
    #        "insert_onto_square_peg"
    #        "meat_on_grill"
    #        "move_hanger"
    #        "open_drawer"
    #        "put_money_in_safe"
    #        "reach_and_drag"
    #        "scoop_with_spatula"
    #        "setup_chess"
    #        "slide_block_to_target"
    #        "stack_cups"
    #        "straighten_rope"
    #        "turn_oven_on"
    #        "wipe_desk")

    tasks=(
           "pick_and_lift_small"
           "pick_up_cup"
           "place_cups"
           "place_hanger_on_rack"
           "place_shape_in_shape_sorter"
           "play_jenga"
           "beat_the_buzz"
           "change_channel"
           "change_clock"
           "close_door"
           "close_drawer"
           "close_fridge"
           "close_grill"
           "close_microwave"
           "hang_frame_on_hanger"
           "hit_ball_with_queue"
           "insert_usb_in_computer"
           "lamp_off"
           "lamp_on"
           "lift_numbered_block"
           "light_bulb_in"
           "light_bulb_out"
           )
else
    echo "Collectins demos from task $1"
    tasks=("$1")
fi

for task in "${tasks[@]}"
do
    python examples/ex_failgen_sweeper.py --task $task \
                --max_tries 20 \
                --headless
done
