# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import List, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "..", "data")
DATA_FILEPATH = os.path.join(DATA_DIR, "data.json")

STR1_BEFORE = (
    ", and then explain the reason why the current sub-tasks has failed."
)
STR1_AFTER = (
    ', and if the answer is "no" explain the reason for the justification'
    + " and using that rational to think step by step and followed by choosing"
    + " your choice from the keys in the MCQ dictionary for the reasons of"
    + " failure, where the MCQ dictionary is given as follows: {mcq}"
)

MCQ_DICTIONARY = {
    "A": "The robot gripper rotated with an incorrect roll angle",
    "B": "The robot gripper move to the desired position with an offset along the x direction",
    "C": "The robot gripper move to the desired position with an offset along the y direction",
    "D": "The robot gripper move to the desired position with an offset along the z direction",
    "E": "The robot gripper fails to close the gripper.",
    "F": "The robot gripper rotated with an incorrect pitch angle",
    "G": "The robot gripper rotated with an incorrect yaw angle",
    "H": "The robot slip the object out of its gripper",
    "I": "The robot execute the action on the wrong object",
    "J": "There is no rotation for the robot gripper",
    "K": "This is not the right action sequence for the task.",
}

MCQ_DICTIONARY_STR = json.dumps(MCQ_DICTIONARY)


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

ANSWERS_MCQ = {
    "grasp": "E",
    "slip": "H",
    "rotation_x": "A",
    "rotation_y": "F",
    "rotation_z": "G",
    "translation_x": "B",
    "translation_y": "C",
    "translation_z": "D",
    "no_rotation": "J",
    "wrong_sequence": "K",
    "wrong_object": "I",
}

BASE_PATH = "/lustre/fsw/portfolios/nvr/users/jiafeid/dataset/test_data/data/"


@dataclass
class Question:
    question_id: int
    image: str
    text: str
    category: str


@dataclass
class Answer:
    question_id: int
    text: str
    category: str


def find_pattern(req_str: str, req_patterns: List[str]) -> Tuple[str, int, int]:
    for failtype_pattern in req_patterns:
        match = re.search(re.escape(failtype_pattern), req_str)
        if match:
            return failtype_pattern, match.start(), match.end()
    return "None", -1, -1


def find_number(req_str: str) -> int:
    match = re.search(r"_wp(\d+)_", req_str)
    if match:
        wp_number = int(match.group(1))
        return wp_number
    return 0


def q_filter_image_name(q_image: str) -> str:
    return os.path.join(BASE_PATH, os.path.basename(q_image))


def q_filter_text(q_text: str) -> str:
    q_text = re.sub(r"<image>\n\s*", "", q_text)
    q_text = re.sub(
        STR1_BEFORE, STR1_AFTER.format(mcq=MCQ_DICTIONARY_STR), q_text
    )
    q_text += '. For example: "no, Z"'

    return q_text


def a_build_answer(a_text: str, image_name: str) -> str:
    image_name = os.path.basename(image_name)
    fail_type, _, _ = find_pattern(image_name, FAILURE_TYPES)
    # fail_wp_idx = find_number(image_name)
    if a_text.startswith("No"):
        a_text = f"No, {MCQ_DICTIONARY[ANSWERS_MCQ[fail_type]]}"

    return a_text


def main() -> int:
    with open(DATA_FILEPATH, "r") as fhandle:
        data_json = json.load(fhandle)

    questions: List[Question] = []
    answers: List[Answer] = []
    for q_json in data_json:
        question = Question(
            question_id=int(q_json["id"]),
            image=q_filter_image_name(q_json["image"]),
            text=q_filter_text(q_json["conversations"][0]["value"]),
            category="detail",
        )
        questions.append(question)

        answer = Answer(
            question_id=int(q_json["id"]),
            text=a_build_answer(
                q_json["conversations"][1]["value"], q_json["image"]
            ),
            category="detail",
        )
        answers.append(answer)

        # Update the json entries in the json data
        q_json["image"] = question.image
        q_json["conversations"][0]["value"] = question.text
        q_json["conversations"][1]["value"] = answer.text

    with open("qa_failgen.json", "w") as fhandle:
        fhandle.write(json.dumps(data_json, indent=4))

    with open("qa_failgen_questions.json", "w") as fhandle:
        for qtn in questions:
            fhandle.write(json.dumps(asdict(qtn)) + "\n")

    with open("qa_failgen_answers.json", "w") as fhandle:
        for ans in answers:
            fhandle.write(json.dumps(asdict(ans)) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
