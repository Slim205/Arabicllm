import fire
from vllm import LLM,SamplingParams
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer
import torch
# Maybe change the instruction by given 5 different starts to the llm 
def get_query_answer(line):
    text = line["text"]

    query = f"""
Does the following statement incite hatred?

'{text}'
    """
    if (line["toxicity_ai"] + line["toxicity_human"] ) > 5.5:
        answer = 'yes'
    else :
        answer = 'no'
    return query,answer

def toxigen(model_name: str, repo_name: str):
    dataset = load_dataset("toxigen/toxigen-data")
    dataset = dataset['train'].select(range(50))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def get_question(sample) : 
        prompt = f"""
Transform the following query to a question : 

{get_query_answer(sample)[0]}

Generate the question directly, without adding any tags, comments, or references to the input text.
"""

        messages = [{"role": "user", "content":prompt}]
        prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
        return prompt_with_chat_template

    def get_answer(sample) : 
        prompt = f"""
Respond to this query with a brief justification (one sentence):

Query: {get_query_answer(sample)[0]}
Answer: {get_query_answer(sample)[1]}

Generate the response directly, without adding any tags, comments, or references to the input text.
"""


        messages = [{"role": "user", "content":prompt}]
        prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
        return prompt_with_chat_template

    prompts =[]
    for example in dataset:        
        prompts.append(get_question(example))
        prompts.append(get_answer(example))

    print(prompts[0])
    llm = LLM(model_name, dtype=torch.float16,max_model_len=1024,gpu_memory_utilization=0.7) 
    sampling_params = SamplingParams(max_tokens=512,temperature=0.8, top_p=0.95)
    outputs = llm.generate(prompts,sampling_params)

    llm_answers=[]
    llm_questions=[]
    for i, item in enumerate(outputs):

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
    fire.Fire(toxigen)
