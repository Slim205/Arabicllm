import fire
from vllm import LLM,SamplingParams
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer
import re
import random
import torch

def get_correct_option(sample) :
    options_list = sample['choices']
    return options_list[sample['answer']]

def get_index(s) :
    return chr(s+ord('A'))

def extract_generated_question(output: str) -> str:
    # Define the regex pattern to match the content between <generated_question> tags
    pattern = r"<generated_question>(.*?)</generated_question>"

    # Use re.search to find the match
    match = re.search(pattern, output, re.DOTALL)

    if match:
        # Return the content between the tags
        return match.group(1).strip()
    else:
        return None

def template(sample) :
    query = f"""
{sample['question']}

A) {sample['choices'][0]} 
B) {sample['choices'][1]}
C) {sample['choices'][2]}
D) {sample['choices'][3]} 
"""
    sample['ift_instruction'] = query
    return sample

def extract_generated_answer(output: str) -> str:
    # Define the regex pattern to match the content between <generated_answer> tags
    pattern = r"<generated_answer>(.*?)</generated_answer>|<generated_answer>(.*)"
    
    # Use re.search to find the match
    match = re.search(pattern, output, re.DOTALL)
    
    if match:
        # Return the content between the tags, handling cases with or without the closing tag
        return (match.group(1) or match.group(2)).strip()
    else:
        # If no match found, return None
        return None


def mmlu(model_name: str, repo_name: str,output_path: str = './mmlu'):

    dataset = load_dataset("cais/mmlu", "all")
    dataset = dataset['auxiliary_train']

    random.seed(42)  # Optional: for reproducibility
    dataset_shuffled = dataset.shuffle(seed=42)
    dataset = dataset_shuffled#.select(range(30))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_answer(sample) : 
        prompt = f"""
Answer the following question with a brief justification (one sentence). Start your response with "{get_index(sample['answer'])})" : 

Question : {sample['ift_instruction']}
Answer : {get_correct_option(sample)}

Respond directly, avoiding any tags, comments, or references to the input text.
"""
        prompt = f"""
You are an AI assistant tasked with reformulating a given answer to a question based on a provided text. Here's what you need to do:

First, carefully read the following instruction:
<instruction>
{sample['ift_instruction']}
</instruction>

The correct answer to this question is:
<answer>
{ {get_correct_option(sample)}}
</answer>

Your task is to reformulate this answer by following these guidelines:
   a) Provide a brief justification (one sentence) for why this answer is correct.
   b) Even if you believe the answer might be incorrect based on the support text, defend it as if it were true.
   c) Do not mention any doubts about the answer's correctness.

Format your response as follows:
   <generated_answer>
    {get_index(sample['answer'])}) [reformulated answer]
   </generated_answer>
"""
        messages = [{"role": "user", "content":prompt}]
        prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
        return prompt_with_chat_template

    if model_name == "meta-llama/Meta-Llama-3.1-70B-Instruct" : 
        llm = LLM(model_name, tensor_parallel_size=4,max_model_len=2048,gpu_memory_utilization=0.93) 

    else :
        #llm = LLM(model_name, dtype=torch.float16,max_model_len=2028)  # To be changed
        llm = LLM(model_name,max_model_len=4096)  # To be changed

    sampling_params = SamplingParams(max_tokens=512,temperature=0.8, top_p=0.95)
    dataset = dataset.map(template)

    prompts = []

    for example in dataset:        
        prompts.append(get_answer(example))

    print(prompts[0])
    outputs = llm.generate(prompts,sampling_params)
    
    llm_answers=[]
    for i, item in enumerate(outputs):
        llm_answers.append(extract_generated_answer(item.outputs[0].text))

    dataset = dataset.add_column("ift_answer", llm_answers)

    dataset_dict = DatasetDict({"train": dataset})

    dataset_dict.save_to_disk(output_path)
    print(f"Translated dataset saved to {output_path}")

    dataset_dict.push_to_hub(repo_name)
    print(f"Translated dataset saved and pushed to Hugging Face repo: {repo_name}")

if __name__ == '__main__':
    fire.Fire(mmlu)
