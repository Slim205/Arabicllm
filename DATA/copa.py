import fire
from vllm import LLM,SamplingParams
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer
import torch

def copa(model_name: str, repo_name: str):

    dataset = load_dataset("stjokerli/TextToText_copa")
    dataset = dataset['train'].select(range(50))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompts = []

    def get_question(sample) : 
        prompt = f"""
Please create a question that incorporates the following premise and two options. The question should be designed to reflect a choice between these two options.

Premise: {sample['premise']}
Options: {sample['choice1']}, {sample['choice2']}

Generate the question directly, without adding any tags, comments, or references to the input text.
"""

        messages = [{"role": "user", "content":prompt}]
        prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
        return prompt_with_chat_template

    def get_sample_answer(sample) :
        if int(sample['label']) == 0 : 
            return sample['choice1']
        if int(sample['label']) == 1 : 
            return sample['choice2']

    def get_answer(sample) : 
        prompt = f"""
Please formulate an answer that addresses the {sample['question']} of the following premise.

Premise: {sample['premise']}
Answer: {get_sample_answer(sample)}

Generate your response directly, without including any tags, comments, or references to the provided text.
"""
        messages = [{"role": "user", "content":prompt}]
        prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
        return prompt_with_chat_template

    for example in dataset:        
        prompts.append(get_question(example))
        prompts.append(get_answer(example))

    print(prompts[0])


    llm = LLM(model_name, dtype=torch.float16,max_model_len=2048) 

    sampling_params = SamplingParams(max_tokens=512,temperature=0.8, top_p=0.95)
    outputs = llm.generate(prompts,sampling_params)
    
    llm_answers=[]
    llm_questions=[]
    for i, item in enumerate(outputs):

        #llm_questions.append(item.outputs[0].text)
        if i % 2 == 0 :
            llm_questions.append(item.outputs[0].text)
        else :
            llm_answers.append(item.outputs[0].text)

    dataset = dataset.add_column("ift_instruction", llm_questions)
    dataset = dataset.add_column("ift_answer", llm_answers)

    dataset_dict = DatasetDict({"train": dataset})

   # dataset_dict.save_to_disk(output_path)
   # print(f"Translated dataset saved to {output_path}")

    dataset_dict.push_to_hub(repo_name)
    print(f"Translated dataset saved and pushed to Hugging Face repo: {repo_name}")

if __name__ == '__main__':
    fire.Fire(copa)
