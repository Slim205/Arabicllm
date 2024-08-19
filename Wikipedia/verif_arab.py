import fire
from vllm import LLM,SamplingParams
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer
import wikipedia as w 

def load_list_from_file(filename):
    with open(filename, 'r') as f:
        data_list = [line.strip() for line in f]
    return data_list

def save_list_to_file(filename, data_list):
    with open(filename, 'w') as f:
        for item in data_list:
            f.write("%s\n" % item)

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def verif_arab():
# Charger les listes depuis les fichiers texte
    titles_level1 = load_list_from_file('titles_level1_filtered_v3.txt')
    
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompts = []

    def get_response(text) : 
        prompt = f"""
Read the following text and determine whether it is related to Arabic language, culture, history, geography, or any relevant Arab country. Your response should start with 'Yes' or 'No':

{text}

Respond directly, avoiding any tags, comments, or references to the input text.

"""

        messages = [{"role": "user", "content":prompt}]
        prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
        return prompt_with_chat_template

    def fetch_summary(title):
        page = w.WikipediaPage(title)
        text = page.summary
        return get_response(text)

    with ThreadPoolExecutor(max_workers=20) as executor:
    # Map the fetch_summary function to titles_level1
        results = list(tqdm(executor.map(fetch_summary, titles_level1), total=len(titles_level1), desc="Fetching Summaries"))

    llm = LLM(model_name,max_model_len=4096) 


    sampling_params = SamplingParams(max_tokens=512,temperature=0.8, top_p=0.95)
    outputs = llm.generate(results,sampling_params)

    llm_answers=[]
    for i, item in enumerate(outputs):
        answer = item.outputs[0].text 
        if 'Yes' in answer : 
            llm_answers.append(titles_level1[i])
    save_list_to_file('titles_filtered_v8.txt', llm_answers)


if __name__ == '__main__':
    verif_arab()