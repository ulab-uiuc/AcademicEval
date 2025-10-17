import os
import json
import pickle
import time
import datetime
import openai
import random
import numpy as np
import torch


def show_time():
    time_stamp = '\033[1;31;40m[' + str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + ']\033[0m'

    return time_stamp


def get_device(index=0):
    return torch.device("cuda:" + str(index) if torch.cuda.is_available() else "cpu")


def text_wrap(text):
    return '\033[1;31;40m' + str(text) + '\033[0m'


def write_to_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


def write_to_pkl(data, output_file):
    with open(output_file, 'wb') as file:
        pickle.dump(data, file)


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def count_entries_in_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return len(data)


def clean_title(title):
    cleaned_title = title.replace("\n", " ").strip()
    cleaned_title = os.path.splitext(cleaned_title)[0]
    cleaned_title = cleaned_title.replace(":", "").replace("- ", " ").replace("-", " ").replace("_", " ").title()

    return cleaned_title


def print_metrics(metrics):
    for k, v in metrics.items():
        ff = "{} " + k + " ("
        metric = metrics[k]
        for sub_k in metric.keys():
            ff += sub_k + "/"
        ff = ff[:-1] + "): "
        for sub_v in metric.values():
            ff += format(sub_v, ".4f") + "/"
        ff = ff[:-1]
        print(ff.format(show_time()))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_llm_response_via_api(prompt,
                             API_BASE="https://api.together.xyz",
                             API_KEY="[YOUR API KEY HERE]",
                             LLM_MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1",
                             TAU=1.0,
                             TOP_P=1.0,
                             N=1,
                             SEED=42,
                             MAX_TRIALS=3,
                             TIME_GAP=3):
    '''
    res = get_llm_response_via_api(prompt='hello')  # Default: TAU Sampling (TAU=1.0)
    res = get_llm_response_via_api(prompt='hello', TAU=0)  # Greedy Decoding
    res = get_llm_response_via_api(prompt='hello', TAU=0.5, N=2, SEED=None)  # Return Multiple Responses w/ TAU Sampling
    '''
    openai.api_base = API_BASE
    openai.api_key = API_KEY
    completion = None
    while MAX_TRIALS:
        MAX_TRIALS -= 1
        try:
            completion = openai.ChatCompletion.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                n=N,
                temperature=TAU,
                top_p=TOP_P,
                seed=SEED,
            )
            break
        except Exception as e:
            print(e)
            print("Retrying...")
            time.sleep(TIME_GAP)

    if completion is None:
        raise Exception("Reach MAX_TRIALS={}".format(MAX_TRIALS))
    contents = completion.choices
    if len(contents) == 1:
        return contents[0].message["content"]
    else:
        return [c.message["content"] for c in contents]


if __name__ == '__main__':
    prompt = "Hello, how are you?"
    res = get_llm_response_via_api(prompt=prompt, TAU=0, LLM_MODEL="mistralai/Mixtral-8x22B-Instruct-v0.1")
    print(res)