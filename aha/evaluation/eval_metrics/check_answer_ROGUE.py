import argparse
import json
from rouge_score import rouge_scorer

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute ROUGE scores from conversation data and answer files."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the JSON file containing conversation data."
    )
    parser.add_argument(
        "--answers_path",
        type=str,
        required=True,
        help="Path to the JSON-lines file containing answers."
    )
    parser.add_argument(
        "--indx_num",
        type=int,
        default=11291,
        help="Number of indices to process (default: 11291)."
    )
    return parser.parse_args()

def compute_rouge(sentence1, sentence2):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(sentence1, sentence2)
    return scores

def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def print_conversation_by_id(data, target_id):
    for item in data:
        if item["id"] == target_id:
            for convo in item["conversations"]:
                if convo["from"] == "gpt":
                    print(convo["value"])
                    # Return the first matching value.
                    return str(convo["value"])
    # Return an empty string if not found.
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
    # Split the string by the first comma
    parts = text.split(',', 1)
    # Check if there is a comma in the text
    if len(parts) > 1:
        # Return the part after the first comma, stripping any leading/trailing whitespace
        return parts[1].strip()
    else:
        # If there is no comma, return an empty string
        return ""

def main():
    args = parse_args()

    data = load_json(args.data_path)
    answers_file = args.answers_path
    indx_num = args.indx_num

    yes_point = 0
    no_point = 0.0
    num_id = []

    for i in range(indx_num):
        print(f"Processing index: {i}")
        target_value = i  # Using the index as target value
        prediction = read_json_line_by_line(answers_file, target_value)
        gt = print_conversation_by_id(data, str(i))

        gt_substring = gt.split(',')[0].strip().lower() if gt else ""
        prediction_substring = prediction.split(',')[0].strip().lower() if prediction else ""

        if gt_substring == "yes" and prediction_substring == "yes":
            yes_point += 1
        elif gt_substring == "no" and prediction_substring == "no":
            new_prediction = extract_words_after_comma(prediction)
            new_GT = extract_words_after_comma(gt)
            print("_____")
            print("Prediction substring:", new_prediction)
            print("GT substring:", new_GT)
            rouge_scores = compute_rouge(new_prediction, new_GT)
            if new_prediction == new_GT:
                num_id.append(i)
            score = rouge_scores['rougeL'][2]
            print("ROUGE-L F1 Score:", score)
            no_point += score

        total_score = (yes_point + no_point) / indx_num
        print("Current F1 score:", total_score)

if __name__ == "__main__":
    main()
