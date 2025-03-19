import json
import os
import anthropic

client = anthropic.Anthropic()
# Define the file paths
gt_path = '/home/jiafeid/rlbench-failgen/evaluation/real_qa.json'
res_path = '/home/jiafeid/rlbench-failgen/evaluation/evaluation/aha_vicuna_real_final_qa_failgen_answers copy.json'

# Load the JSON data
with open(gt_path, 'r') as file:
    gt_data = json.load(file)
with open(res_path, 'r') as file:
    res_data = json.load(file)


import re

def llm_fuzzy_match_claude(pred: str, reference: str, question: str) -> float:
    prompt = (
        "Help a teacher to grade the answer of a student given a question. Keep in mind that the student may use "
        "different phrasing or wording to answer the question. The goal is to evaluate whether the answer is "
        "semantically equivalent to the reference answer.\n"
        f"question: {question}\n"
        f"reference answer: {reference}\n"
        "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
        f"student answer: {pred}\n"
        "Conclude the judgement by stating 'Judgement: correct/incorrect/partially correct.'"
    )
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0,
        system="You are a helpful assistant",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    
    # Extract the 'text' from the message content
    response_text = message.content[0].text if message.content else ""
    print(response_text)

    # Search for "judgement:" and the following word
    match = re.search(r'judgement:\s*(\w+)', response_text, re.IGNORECASE)
    
    if match:
        judgement = match.group(1).lower()
        if judgement == "incorrect":
            print('score:0')
            return 0.0
        elif judgement == "partially":  # Assuming "partially correct" is split into two words
            print('score:0.5')
            return 0.5
        elif judgement == "correct":
            print('score:1')
            return 1.0
    
    print("Warning: No clear judgement found in response, skipping entry.")
    return None
# def llm_fuzzy_match(pred: str, reference: str, question: str) -> float:
#     message = (
#         "Help a teacher to grade the answer of a student given a question. Keep in mind that the student may use "
#         "different phrasing or wording to answer the question. The goal is to evaluate whether the answer is "
#         "semantically equivalent to the reference answer.\n"
#         f"question: {question}\n"
#         f"reference answer: {reference}\n"
#         "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
#         f"student answer: {pred}\n"
#         "Conclude the judgement by correct/incorrect/partially correct.\n"
#     )
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": message},
#     ]

#     response = client_openai.chat.completions.create(
#         model="gpt-4o",
#         messages=messages,
#         temperature=0,
#     )
#     response = response.choices[0].message.content.lower()
#     print(response)
#     if "incorrect" in response:
#         return 0.0
#     elif "partially correct" in response:
#         return 0.5
#     else:
#         if "correct" not in response:
#             print("Warning: 'correct' not found in response, skipping entry.")
#             return None
#         return 1.0

# Ensure both files have the same length
assert len(gt_data) == len(res_data)

gpt_scores = []
for i in range(len(gt_data)):
    print(i)
    gt = gt_data[i]['conversations'][1]['value']
    print(gt)
    res = res_data[i]['text']
    qn = gt_data[i]['conversations'][0]['value']
    print(res)
    
    try:
        gpt_score = llm_fuzzy_match_claude(res, gt, qn)
        if gpt_score is not None:  # Only append valid scores
            gpt_scores.append(gpt_score)
    except Exception as e:
        print(f"Error processing index {i}: {e}, skipping this entry.")

# Calculate and print the rating
if gpt_scores:
    print(len(gpt_scores))
    rating = sum(gpt_scores) / len(gpt_scores)
    print(f"Final Rating: {rating}")
else:
    print("No valid scores to compute rating.")
