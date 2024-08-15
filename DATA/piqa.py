import fire
from vllm import LLM,SamplingParams
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer
import torch

def piqa(model_name: str, repo_name: str,output_path: str = './piqa'):

    dataset = load_dataset("generate_piqa_data.py")
    dataset = dataset['train'].select(range(50))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_question(sample) : 
        prompt = f"""
Please create a question that incorporates the following premise and options, reflecting a choice between the two. Include both options in the question.

Premise: {sample['goal']}
Options: {sample['sol1']}, {sample['sol2']}

Generate the question directly, without adding any tags, comments, or references to the input text.
"""

        messages = [{"role": "user", "content":prompt}]
        prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
        return prompt_with_chat_template

    def get_correct_option(sample) :
        if int(sample['label']) == 0 :
            return sample['sol1']
        else :
            return sample['sol2']
    def get_answer(sample) : 
        prompt = f"""
Answer the following question with a brief justification (one sentence is sufficient):

{sample['ift_instruction']}

(Note: The correct option is {get_correct_option(sample)}.)

Respond directly, avoiding any tags, comments, or references to the input text.
"""
        messages = [{"role": "user", "content":prompt}]
        prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
        return prompt_with_chat_template

    prompts = []

    for example in dataset:        
        prompts.append(get_question(example))

    print(prompts[0])
    llm = LLM(model_name, dtype=torch.float16,max_model_len=2048) 
    sampling_params = SamplingParams(max_tokens=512,temperature=0.8, top_p=0.95)
    outputs = llm.generate(prompts,sampling_params)
    llm_questions=[]
    for i, item in enumerate(outputs):
        llm_questions.append(item.outputs[0].text)

    dataset = dataset.add_column("ift_instruction", llm_questions)


    prompts = []

    for example in dataset:        
        prompts.append(get_answer(example))

    print(prompts[0])
    outputs = llm.generate(prompts,sampling_params)
    
    llm_answers=[]
    for i, item in enumerate(outputs):
        llm_answers.append(item.outputs[0].text)

    dataset = dataset.add_column("ift_answer_intermediate", llm_answers)

    def get_str(s) :
        if int(s)==0:
            return "first"
        return "second"

    def get_rank(sample) : 
        prompt = f"""
Combine these two sentences into a complete response, maintaining their distinct meanings and keeping them separate.

The {get_str(sample['label'])} option is the correct option.
{sample['ift_answer_intermediate']} 

Please respond concisely, without adding any tags, comments, or references to the input.
"""


        messages = [{"role": "user", "content":prompt}]
        prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
        return prompt_with_chat_template

    prompts = []

    for example in dataset:        
        prompts.append(get_rank(example))

    print(prompts[0])
    outputs = llm.generate(prompts,sampling_params)
    llm_answers_rank=[]
    for i, item in enumerate(outputs):
        llm_answers_rank.append(item.outputs[0].text)

    dataset = dataset.add_column("ift_answer", llm_answers_rank)
    dataset = dataset.remove_columns("ift_answer_intermediate")


    dataset_dict = DatasetDict({"train": dataset})

    dataset_dict.save_to_disk(output_path)
    print(f"Translated dataset saved to {output_path}")

    dataset_dict.push_to_hub(repo_name)
    print(f"Translated dataset saved and pushed to Hugging Face repo: {repo_name}")

if __name__ == '__main__':
    fire.Fire(piqa)
