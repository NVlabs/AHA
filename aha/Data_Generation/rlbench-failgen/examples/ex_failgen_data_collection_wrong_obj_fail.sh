#!/usr/bin/env bash

if [ $# -eq 0 ]
  then
    echo "Collecting data for all tasks"

    tasks=("push_buttons"
           "open_jar")
else
    echo "Collectins data for task $1"
    tasks=("$1")
fi

for task in "${tasks[@]}"
do
    python examples/ex_failgen_data_collection.py --task $task \
                --episodes 10 --max_tries 20 --video --failtype wrong_object
done
