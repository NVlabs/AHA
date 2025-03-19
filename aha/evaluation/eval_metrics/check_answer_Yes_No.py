import argparse
import json
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import nltk
import sacrebleu

nltk.download('punkt')

def compute_bleu_score(prediction, ground_truth):
    # sacrebleu expects the ground truth as a list of references
    bleu = sacrebleu.sentence_bleu(prediction, [ground_truth])
    return bleu.score

def compute_rouge(sentence1, sentence2):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(sentence1, sentence2)
    return scores

def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def print_conversation_by_id(data, target_id):
    # Returns the first GPT conversation matching the target_id
    for item in data:
        if item["id"] == target_id:
            for convo in item["conversations"]:
                if convo["from"] == "gpt":
                    print(convo["value"])
                    return str(convo["value"])
    return ""

def read_json_line_by_line(file_path, target_value):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    first_key = 'question_id'
                    first_value = data[first_key]
                    if first_value == target_value:
                        print(data['text'])
                        return str(data['text'])
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                except StopIteration:
                    print("JSON object is empty, no keys to read.")
                except Exception as e:
                    print(f"An error occurred: {e}")
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except IOError as e:
        print(f"An error occurred while reading the file: {e}")
    return ""

def extract_words_after_comma(text):
    # Split the string by the first comma and return the remainder
    parts = text.split(',', 1)
    if len(parts) > 1:
        return parts[1].strip()
    else:
        return ""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions using conversation data and compute F1 scores."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the conversation data JSON file."
    )
    parser.add_argument(
        "--answers_path",
        type=str,
        required=True,
        help="Path to the answers JSON lines file."
    )
    parser.add_argument(
        "--indx_num",
        type=int,
        default=11291,
        help="Number of indices to process (default: 11291)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    data = load_json(args.data_path)
    answers_file = args.answers_path
    indx_num = args.indx_num

    yes_point = 0
    no_point = 0.0
    total_yes = 0

    for i in range(indx_num):
        print(f"Processing index: {i}")
        target_value = i  # Using the index as target value
        prediction = read_json_line_by_line(answers_file, target_value)
        gt = print_conversation_by_id(data, str(i))
        
        gt_substring = gt.split(',')[0].strip().lower() if gt else ""
        if gt_substring == "yes":
            total_yes += 1
        prediction_substring = prediction.split(',')[0].strip().lower() if prediction else ""
        
        if gt_substring == "yes" and prediction_substring == "yes":
            yes_point += 1
        elif gt_substring == "no" and prediction_substring == "no":
            no_point += 1

        # Optionally, you can print a running F1 score here using the current index + 1.
        current_f1 = (yes_point + no_point) / (i + 1)
        print(f"Current F1 score: {current_f1}")

    final_f1 = (yes_point + no_point) / indx_num
    print("Final F1 score:", final_f1)

if __name__ == "__main__":
    main()
