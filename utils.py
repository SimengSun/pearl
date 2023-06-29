import openai
import csv
import os
import pickle
import tiktoken
import random
import time


enc = tiktoken.get_encoding("p50k_base")
model_name = "gpt-4-0314" #"gpt-4" #
max_output_len = 512
option_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
map_prompt = "Relevant information for answering the question:\n\n{open_answer}\n\nQuestion:{question}\n{options}\n\nRead the relevant information about the article and answer the question by selecting the best option above. Only one of them is correct.\n\nAnswer (select from A, B, C, D):\n"

def openai_setapi():
    print("set open_ai api")
    openai.api_type = "open_ai"
    openai.api_base = 'https://api.openai.com/v1'
    openai.api_version = None
    openai.api_key = os.getenv("OPENAI_API_KEY_OAI")

def get_response_openai(kwargs):
    run = True
    skip = False
    cnt = 0
    jitter = True
    while run:
        try:
            response = openai.ChatCompletion.create(**kwargs)
            os.system("sleep 1s")
            if len(response["choices"][0]["message"]["content"].strip()) > 0:
                run = False
            else:
                cnt += 1
                if cnt >= 1:
                    skip, run = True, False

        except Exception as e:
            if "Rate limit" in str(e):
                delay = 5 + jitter * random.random()
                time.sleep(delay)
                cnt += 1
                if cnt % 100 == 0:
                    print(f"retried count: {cnt}")
                    print(e)
            else:
                print(e)
                run = False
                skip = True
    if skip:
        return None
    
    return response["choices"][0]["message"]["content"].lstrip().rstrip()

def get_response(prompt, model="gpt-4-0314", 
                max_tokens=512, temperature=0.0, top_p=0.0,
                frequency_penalty=0, 
                presence_penalty=0, 
                stop=None): 
    
    kwargs = {
        'model': model,
        'messages': [{"role":"user", "content": prompt}],
        'temperature': temperature,
        'max_tokens': max_tokens+1,
        'top_p': top_p,
        'frequency_penalty': frequency_penalty,
        'presence_penalty': presence_penalty,
        'stop': stop
    }
    
    ret = get_response_openai(kwargs)
    return ret

def load_questions(fname):
    with open(fname, "r") as f:
        reader = csv.DictReader(f)
        questions = {}
        for row in reader:
            questions[row['qid']] = row
    return questions

def load_question_filter():
    with open("./data/processed/qid_filter.pkl", "rb") as f:
        return pickle.load(f)

def process_article(article, chunk_size=512):
    return article.replace("\n\n", "##@@").replace("\n", " ").replace("##@@", "\n\n")

def load_text(fname):
    if fname is None:
        raise ValueError("fname is None")
    with open(fname, "r") as f:
        return f.read()

def load_prompt(fname):
    if fname is None:
        raise ValueError("fname is None")
    with open(fname, "r") as f:
        return f.read()

def  load_pickle(fname):
    if fname is None:
        raise ValueError("fname is None")
    with open(fname, "rb") as f:
        return pickle.load(f)

