#!/usr/bin/env bash

if [ $# -eq 0 ]
  then
    echo "Collecting demos for all tasks"


    tasks=(
        "stack_chairs"
        "place_hanger_on_rack"
        "place_cups"
        "light_bulb_out"
        "light_bulb_in"
        "lamp_on"
        "hockey"
        "open_oven"
        "meat_on_grill"
        "pick_up_cup"
    )

else
    echo "Collectins demos from task $1"
    tasks=("$1")
fi

for task in "${tasks[@]}"
do
    xvfb-run -a -s "-screen 0 1400x900x24" python examples/ex_custom_data_generator.py \
         --task $task --max-tries 10 --headless --num-episodes 100
done