import os
import re
import argparse
import yaml
import random
import json

def parse_folder_name(folder_name):
    """
    Parse the folder name assuming it follows the pattern:
      <task>_<failure>_wp<number>_episode<number>
      
    The failure mode must be one of the following:
      grasp, slip, translation_x, translation_y, translation_z,
      rotation_x, rotation_y, rotation_z, no_rotation, wrong_sequence

    Example:
      basketball_in_hoop_grasp_wp1_episode0
      
    Returns:
      (task_name, failure_mode, failure_frame, episode)
    """
    valid_failure_modes = [
        "grasp", "slip", "translation_x", "translation_y", "translation_z",
        "rotation_x", "rotation_y", "rotation_z", "no_rotation", "wrong_sequence"
    ]
    failure_mode_pattern = "(" + "|".join(valid_failure_modes) + ")"
    pattern = r"^(.*)_" + failure_mode_pattern + r"_wp(\d+)_episode(\d+)$"
    
    match = re.match(pattern, folder_name)
    if match:
        task_name = match.group(1)
        failure_mode = match.group(2)
        failure_frame = int(match.group(3))
        episode = int(match.group(4))
        return task_name, failure_mode, failure_frame, episode
    else:
        raise ValueError("Folder name does not match the expected pattern or contains an invalid failure mode.")

def frame_index(filename):
    """
    Extract the integer index from a filename like '0.png'
    """
    base, _ = os.path.splitext(filename)
    try:
        return int(base)
    except ValueError:
        return -1  # In case the filename doesn't follow the expected numeric pattern

def load_sub_tasks(yaml_file_path):
    """
    Load the sub-tasks from the YAML file.
    The YAML file is expected to have a structure with a "sub-tasks" key that is a list.
    Each sub-task item should have a "task_no" and a "task_description".
    
    Returns:
      A mapping from task_no (int) to the sub-task item.
    """
    with open(yaml_file_path, "r") as yf:
        try:
            data = yaml.safe_load(yf)
            sub_tasks_list = data.get("sub-tasks", [])
            sub_tasks_map = {}
            for item in sub_tasks_list:
                task_no = item.get("task_no")
                if task_no is not None:
                    sub_tasks_map[int(task_no)] = item
            return sub_tasks_map
        except Exception as e:
            raise ValueError(f"Error loading sub-tasks YAML file: {e}")

def label_frames(folder_path, sub_tasks_map):
    """
    Process image frames within the folder, label each frame based on the failure frame,
    and attach the corresponding task description from the provided sub_tasks_map.
    
    Rules:
      - Frames with index less than the failure frame are labeled as 'Yes'
      - Frames with index greater than or equal to the failure frame are labeled as 'No'
      
    Additionally, a "question" label is generated for each frame using the provided template.
    The template substitutes [task_description] with the corresponding value from the sub-tasks.
    
    A new "answer" label is added:
      - If label is "Yes", answer = "Yes, the robot succeed at the sub-task."
      - If label is "No", answer = "No, the robot failed because of <failure_mode>."
    """
    folder_name = os.path.basename(folder_path.rstrip(os.sep))
    task_name, failure_mode, failure_frame, episode = parse_folder_name(folder_name)

    # Get list of .png files and sort them numerically.
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    png_files.sort(key=frame_index)

    labels = {}
    for f in png_files:
        index = frame_index(f)
        label = "Yes" if index < failure_frame else "No"
        
        # Determine the answer based on label.
        if label == "Yes":
            answer = f"{label}, the robot succeed at the sub-task."
        else:
            answer = f"{label}, the robot failed because of {failure_mode}."
        
        # Retrieve sub-task info using frame index as task_no.
        task_description = None
        if index in sub_tasks_map:
            sub_task = sub_tasks_map[index]
            descriptions = sub_task.get("task_description")
            if isinstance(descriptions, list) and descriptions:
                task_description = random.choice(descriptions)
            elif isinstance(descriptions, str):
                task_description = descriptions

        # Construct the full image path.
        full_path = os.path.join(folder_path, f)
        
        # Prepare task description text for the question (use empty string if None).
        task_description_text = task_description if task_description is not None else ""
        
        # Create the question label using the provided template.
        question = (
            f"<image>\n"
            f"In the image, it contains many frames from different views at different time steps "
            f"denoted in annotated numbers on each cell top left in a matrix configuration with depicting "
            f"the robot arm performing {task_description_text}. "
            f"The frames are combined with different views, from top to bottom: the top row is the front view; "
            f"the middle row is the wrist view, and the bottom is the overhead view. "
            f"Each frame is labeled with the corresponding timestep reflecting the temporal information from left to right. "
            f"The white grid represents future sub-tasks that has yet to happen and should be ignore. "
            f"For the given sub-tasks, first determine it has succeed by choosing from [\"yes\", \"no\"], and then explain the reason why the current sub-tasks has failed."
        )
        
        labels[f] = {
            "task": task_name,
            "failure_mode": failure_mode,
            "frame_index": index,
            "label": label,
            "answer": answer,
            "task_description": task_description,
            "image_path": full_path,
            "question": question
        }
    return labels

def main():
    parser = argparse.ArgumentParser(
        description="Label frames in sub-folders based on folder name pattern and attach task descriptions from YAML files."
    )
    parser.add_argument("input_dir", help="Path to the directory containing sub-folders with frames and encoded name information.")
    parser.add_argument("--output", default="output.json", help="Path to save the output JSON file.")
    args = parser.parse_args()

    input_directory = args.input_dir
    output_file = args.output

    if not os.path.isdir(input_directory):
        print(f"Input directory '{input_directory}' does not exist or is not a directory.")
        return

    final_output = []
    global_counter = 0

    # Process each sub-folder within the input directory.
    folder_names = [d for d in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, d))]
    for folder in folder_names:
        folder_path = os.path.join(input_directory, folder)
        try:
            task_name, failure_mode, failure_frame, episode = parse_folder_name(folder)
        except Exception as e:
            print(f"Skipping folder '{folder}': {e}")
            continue

        # Construct the YAML file path based on the task name.
        # Adjust the parent_dir path if necessary.
        parent_dir = '/net/nfs/prior/jiafei/AHA-main/aha/Data_Generation/rlbench-failgen/failgen/configs'
        yaml_file_path = os.path.join(parent_dir, f"{task_name}.yaml")
        if not os.path.exists(yaml_file_path):
            print(f"YAML file '{yaml_file_path}' does not exist for folder '{folder}'. Skipping.")
            continue

        try:
            sub_tasks_map = load_sub_tasks(yaml_file_path)
        except Exception as e:
            print(f"Error loading YAML for folder '{folder}': {e}")
            continue

        try:
            labels = label_frames(folder_path, sub_tasks_map)
        except Exception as e:
            print(f"Error processing folder '{folder}': {e}")
            continue

        # Generate JSON objects for each frame in the folder.
        sorted_files = sorted(labels, key=lambda f: frame_index(f))
        for filename in sorted_files:
            global_counter += 1
            frame_label = labels[filename]
            json_obj = {
                "id": str(global_counter),
                "image": frame_label["image_path"],
                "conversations": [
                    {
                        "from": "human",
                        "value": frame_label["question"]
                    },
                    {
                        "from": "gpt",
                        "value": frame_label["answer"]
                    }
                ]
            }
            final_output.append(json_obj)

    # Save the combined JSON to the specified output file.
    with open(output_file, "w") as out_f:
        json.dump(final_output, out_f, indent=4)
    print(f"JSON output saved to '{output_file}'.")

if __name__ == "__main__":
    main()
