import fire
from vllm import LLM,SamplingParams
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer
import re

def get_str(s) :
    if int(s)==0:
        return "first"
    return "second"

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

def piqa(model_name: str, repo_name: str,output_path: str = './piqa'):

    dataset = load_dataset("generate_piqa_data.py")
    dataset = dataset['train']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_question(sample) : 
        prompt = f"""
You are tasked with creating a question that incorporates a given premise and two options, reflecting a choice between them. Follow these instructions carefully:

1. You will be provided with two input variables:
   <OPTIONS>
   {sample['sol1']}
   {sample['sol2']}
   </OPTIONS>

   <PREMISE>
   {sample['goal']}
   </PREMISE>

2. Create a question that:
   - Incorporates the given premise
   - Includes both options as choices
   - Maintains the order of the options as provided

3. Format your generated question as follows:
   <generated_question>
   [Your question here]
   
   1. [Option 1]
   2. [Option 2]
   </generated_question>

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
Answer the following question with a brief justification (one sentence). Start your response with "The {get_str((sample['label']))} option is correct." : 

Question : {sample['ift_instruction']}
Answer : {get_correct_option(sample)}

Respond directly, avoiding any tags, comments, or references to the input text.
"""
        messages = [{"role": "user", "content":prompt}]
        prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
        return prompt_with_chat_template

    prompts = []

    for example in dataset:        
        prompts.append(get_question(example))

    print(prompts[0])
    if model_name == "meta-llama/Meta-Llama-3.1-70B-Instruct" : 
        llm = LLM(model_name, tensor_parallel_size=4,max_model_len=2048,gpu_memory_utilization=0.93) 

    else :
        llm = LLM(model_name, max_model_len=4096) 
    sampling_params = SamplingParams(max_tokens=512,temperature=0.8, top_p=0.95)
    outputs = llm.generate(prompts,sampling_params)
    llm_questions=[]
    for i, item in enumerate(outputs):
        llm_questions.append(extract_generated_question(item.outputs[0].text))

    dataset = dataset.add_column("ift_instruction", llm_questions)


    prompts = []

    for example in dataset:        
        prompts.append(get_answer(example))

    print(prompts[0])
    outputs = llm.generate(prompts,sampling_params)
    
    llm_answers=[]
    for i, item in enumerate(outputs):
        llm_answers.append(item.outputs[0].text)

    dataset = dataset.add_column("ift_answer", llm_answers)

    dataset_dict = DatasetDict({"train": dataset})

    dataset_dict.save_to_disk(output_path)
    print(f"Translated dataset saved to {output_path}")

    dataset_dict.push_to_hub(repo_name)
    print(f"Translated dataset saved and pushed to Hugging Face repo: {repo_name}")

if __name__ == '__main__':
    fire.Fire(piqa)
