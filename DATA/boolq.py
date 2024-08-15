import fire
from vllm import LLM,SamplingParams
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer
import torch

def get_instruction(question, text):
    instruction = f"""
    « {text} »

    {question}
    """
    return instruction


def boolq(model_name: str, repo_name: str,output_path: str = './boolq'):

    dataset = load_dataset("google/boolq")
    dataset = dataset['train'].select(range(50))
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    def get_answer(sample) : 
        prompt = f"""
Read the text below and answer the following question with a brief justification (one sentence). Start your response with "{str(sample['answer'])}," : 

Text: {sample['passage']}
Question: {sample['question']}
Answer: {sample['answer']}

Generate your response without including any tags, comments, or references to the provided text.
"""
        messages = [
            {"role": "user", "content":prompt},

        ]
        prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
        return prompt_with_chat_template

    prompts = []
    llm_questions = []
    for example in dataset:        
        prompts.append(get_answer(example))
        llm_questions.append(get_instruction(example['question'],example['passage']) )
    print(prompts[0])

    llm = LLM(model_name, dtype=torch.float16,max_model_len=2048) 

    sampling_params = SamplingParams(max_tokens=512,temperature=0.8, top_p=0.95)
    outputs = llm.generate(prompts,sampling_params)

    llm_answers=[]
    for i, item in enumerate(outputs):
        llm_answers.append(item.outputs[0].text)

    dataset = dataset.add_column("ift_instruction", llm_questions)
    dataset = dataset.add_column("ift_answer", llm_answers)

    dataset_dict = DatasetDict({"train": dataset})

    dataset_dict.save_to_disk(output_path)
    print(f"Translated dataset saved to {output_path}")

    dataset_dict.push_to_hub(repo_name)
    print(f"Translated dataset saved and pushed to Hugging Face repo: {repo_name}")

if __name__ == '__main__':
    fire.Fire(boolq)
