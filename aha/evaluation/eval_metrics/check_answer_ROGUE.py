import json
from rouge_score import rouge_scorer

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
    return str(convo["value"])
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
# data = load_json('/home/jiafeid/rlbench-failgen/evaluation/valdata.json')
# data = load_json('/home/jiafeid/rlbench-failgen/evaluation/outman_qa2.json')
data = load_json('/home/jiafeid/rlbench-failgen/evaluation/out_qa.json')
# data = load_json('/home/jiafeid/rlbench-failgen/evaluation/real_qa.json')


yes_point=0
# Example usage
# file_path = "/home/jiafeid/rlbench-failgen/evaluation/evaluation/Aha_13B_34k_out_qa_failgen_answers.json"
file_path = 'evaluation/evaluation/aha_arnold_out_final_qa_failgen_answers.json'
no_point=0.0
total_yes=0
# indx_num=138
# indx_num=6868

indx_num=11291
# indx_num=56
num_id=[]
for i in range(indx_num):
    id_num= i
    print(i)
    target_value = id_num # Replace this with the value you're looking for
    prediction= read_json_line_by_line(file_path, target_value)
    gt= print_conversation_by_id(data, str(id_num))

    gt_substring = gt.split(',')[0].strip().lower()
    prediction_substring = prediction.split(',')[0].strip().lower()

    if gt_substring == "yes" and prediction_substring == "yes":
        yes_point +=1
    elif gt_substring == "no" and prediction_substring == "no":
        new_prediction = extract_words_after_comma(prediction)
        new_GT = extract_words_after_comma(gt)
        print("_____")
        print(new_prediction)
        print(new_GT)
        rouge_scores = compute_rouge(new_prediction, new_GT)
        if str(new_prediction)==str(new_GT):
            num_id.append(i)
        # print("ROUGE-L:", rouge_scores['rougeL'])
        score= rouge_scores['rougeL'][2]
        print("ROUGE-L F1 Score:", score)
        no_point += score



    # print("GT:", gt_substring)
    # print("Prediction:", prediction_substring)
    print("F1 score:", (yes_point+no_point)/indx_num)
# print(num_id)