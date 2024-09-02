import fire
from vllm import LLM,SamplingParams
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer



def get_answer(sample,tokenizer) : 
    prompt = f"""
Reformulate the following answer, simplifying the mathematical structure:

Answer: {sample['answer']}
Question: {sample['question']}

Respond directly, avoiding any tags, comments, or references to the original text.
"""

    messages = [{"role": "user", "content":prompt}]
    prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
    return prompt_with_chat_template


def math(model_name: str, repo_name: str,output_path: str = './gsm8k'):

    dataset = load_dataset("openai/gsm8k",'main')
    dataset = dataset['train']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_name == "google/gemma-2-27b-it" :
        llm = LLM(model_name,max_model_len=4096, tensor_parallel_size=2,gpu_memory_utilization=0.85)
    elif model_name == "meta-llama/Meta-Llama-3.1-70B-Instruct" :
        llm = LLM(model_name,max_model_len=4096, tensor_parallel_size=4,gpu_memory_utilization=0.93)

    else :
        llm = LLM(model_name,max_model_len=4096) 
    sampling_params = SamplingParams(max_tokens=512,temperature=0.8, top_p=0.95)


    prompts = []

    for example in dataset:        
        prompts.append(get_answer(example,tokenizer))

    print(prompts[0])
    outputs = llm.generate(prompts,sampling_params)
    llm_answers=[]
    llm_questions=[]
    for i, item in enumerate(outputs):
        llm_answers.append(item.outputs[0].text)
        llm_questions.append(dataset['question'][i])

    dataset = dataset.add_column("ift_instruction", llm_questions)
    dataset = dataset.add_column("ift_answer", llm_answers)


    dataset_dict = DatasetDict({"train": dataset})

    dataset_dict.save_to_disk(output_path)
    print(f"Translated dataset saved to {output_path}")

    dataset_dict.push_to_hub(repo_name)
    print(f"Translated dataset saved and pushed to Hugging Face repo: {repo_name}")

if __name__ == '__main__':
    fire.Fire(math)