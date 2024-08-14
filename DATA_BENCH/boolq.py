import fire
from vllm import LLM,SamplingParams
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer
import torch

def boolq(model_name: str, repo_name: str):

    dataset = load_dataset("google/boolq")
    dataset = dataset['train'].select(range(50))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompts = []

    def apply_chat_template(sample) : 

        prompt = f"""
Please read the text below and provide a more detailed response to the question that follows, without mentioning or referring to the original text:

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

    llm_questions = []

    for example in dataset:        
        prompts.append(apply_chat_template(example))
        llm_questions.append(example['question'])

    print(prompts[0])

    if model_name == "google/gemma-2-27b-it" :
        llm = LLM(model_name, dtype=torch.float16,tensor_parallel_size=2,max_model_len=2048,gpu_memory_utilization=0.8) 
    else : 
        llm = LLM(model_name, dtype=torch.float16,max_model_len=2048) 

    sampling_params = SamplingParams(max_tokens=512,temperature=0.8, top_p=0.95)
    outputs = llm.generate(prompts,sampling_params)

    llm_answers=[]
    for i, item in enumerate(outputs):
        llm_answers.append(item.outputs[0].text)
    
    data = { "question": llm_questions, "answer": llm_answers}
    dataset = Dataset.from_dict(data)
    dataset_dict = DatasetDict({"train": dataset})

   # dataset_dict.save_to_disk(output_path)
   # print(f"Translated dataset saved to {output_path}")

    dataset_dict.push_to_hub(repo_name)
    print(f"Translated dataset saved and pushed to Hugging Face repo: {repo_name}")

if __name__ == '__main__':
    fire.Fire(boolq)
