import fire
from vllm import LLM,SamplingParams
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer



def acva(model_name: str, repo_name: str,output_path: str = './acva'):

    dataset = load_dataset("Slim205/new_acva")
    dataset = dataset['train']
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    def get_answer(sample) : 
        prompt = f"""
Justify in one sentence why the following statement is {sample['answer']}.

statement: {sample['statement']}

Generate your response without including any tags, comments, or references to the provided text.
"""
        messages = [
            {"role": "user", "content":prompt},

        ]
        prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
        return prompt_with_chat_template

    prompts = []
    for example in dataset:        
        prompts.append(get_answer(example))
    print(prompts[0])

    llm = LLM(model_name, tensor_parallel_size=4,max_model_len=2048,gpu_memory_utilization=0.93) 

    sampling_params = SamplingParams(max_tokens=128,temperature=0.8, top_p=0.95)
    outputs = llm.generate(prompts,sampling_params)

    llm_answers=[]
    for i, item in enumerate(outputs):
        if dataset[i]['answer'] : 
            answer = 'True : '
        else :
            answer = 'False : '
        output = answer+ item.outputs[0].text
        llm_answers.append(output)

    dataset = dataset.add_column("ift_answer", llm_answers)

    dataset_dict = DatasetDict({"train": dataset})

    dataset_dict.save_to_disk(output_path)
    print(f"Translated dataset saved to {output_path}")

    dataset_dict.push_to_hub(repo_name)
    print(f"Translated dataset saved and pushed to Hugging Face repo: {repo_name}")

if __name__ == '__main__':
    fire.Fire(acva)
