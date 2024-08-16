import fire
from vllm import LLM,SamplingParams
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer
import torch

def get_str(s) :
    if int(s)==0:
        return "first"
    elif  int(s)==1 :
        return "second"
    elif int(s) == 2 :
        return "third"
    else :
        return "last"

def get_instruction(question, text):
    instruction = f"""
    « {text} »

    {question}
    """
    return instruction


def hellaswag(model_name: str, repo_name: str,output_path: str = './hellaswag'):

    dataset = load_dataset("Rowan/hellaswag")
    dataset = dataset['train'].select(range(50))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_question(sample) : 
        prompt = f"""
Please create a question that incorporates the following context and options, reflecting a choice between the options. Include all options in the question.

Options: {sample['endings'][0]}, {sample['endings'][1]}, {sample['endings'][2]}, {sample['endings'][3]}
Context: {sample['ctx']}

Generate the question directly, without adding any tags, comments, or references to the input text.
"""

        messages = [{"role": "user", "content":prompt}]
        prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
        return prompt_with_chat_template

    def get_correct_option(sample) :
        index = int(sample['label'])
        return sample['endings'][index]
    def get_answer(sample) : 
        prompt = f"""
Answer the following question with a brief justification (one sentence). Start your response with "The {get_str(sample['label'])} option is correct." : 

{sample['ift_instruction']}
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
    llm = LLM(model_name, dtype=torch.float16,max_model_len=2048) 
    sampling_params = SamplingParams(max_tokens=512,temperature=0.8, top_p=0.95)
    outputs = llm.generate(prompts,sampling_params)
    llm_questions=[]
    for i, item in enumerate(outputs):
        llm_questions.append(get_instruction(item.outputs[0].text, dataset['ctx_a'][i]))

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
    fire.Fire(hellaswag)
