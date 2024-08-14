import fire
from vllm import LLM,SamplingParams
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer
import torch
def give_answer(id,options) :
    index = ord(id)-ord('A')
    return options[index]

def get_question_template(sample) : 
    prompt = f"""
Read the following text and answer the question provided:

Text: {sample['article']}
Question: {sample['question']}
"""
    return prompt
def race(model_name: str, repo_name: str):

    dataset = load_dataset("ehovy/race","all")
    dataset = dataset['train'].select(range(50))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompts = []

    def apply_chat_template(sample) : 

        prompt = f"""
Please read the text below and provide a more detailed response to the question that follows, without mentioning or referring to the original text:

Text: {sample['article']}
Question: {sample['question']}
Answer: {give_answer(sample['answer'],sample['options'])}

Generate your response without including any tags, comments, or references to the provided text.
"""
        messages = [
            {"role": "user", "content":prompt},

        ]
        prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
        return prompt_with_chat_template

    llm_questions = []

    for example in dataset:        
        prompts.append(apply_chat_template(example))
        llm_questions.append(get_question_template(example))

    print(prompts[0])


    llm = LLM(model_name, dtype=torch.float16,max_model_len=2048) 

    sampling_params = SamplingParams(max_tokens=512,temperature=0.8, top_p=0.95)
    outputs = llm.generate(prompts,sampling_params)

    llm_answers=[]
    for i, item in enumerate(outputs):
        llm_answers.append(item.outputs[0].text)
    
    dataset = dataset.add_column("ift_answer", llm_answers)
    dataset = dataset.add_column("ift_questions", llm_questions)

    dataset_dict = DatasetDict({"train": dataset})

   # dataset_dict.save_to_disk(output_path)
   # print(f"Translated dataset saved to {output_path}")

    dataset_dict.push_to_hub(repo_name)
    print(f"Translated dataset saved and pushed to Hugging Face repo: {repo_name}")

if __name__ == '__main__':
    fire.Fire(race)
