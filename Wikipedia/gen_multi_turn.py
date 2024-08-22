import fire
from vllm import LLM,SamplingParams
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer



def get_question1(sample,tokenizer) : 
    prompt = f"""
Read the text below and create an open question that the text can answer.

{sample['passage']}

Provide only the question, without any tags, comments, or references to the input text.
"""

    messages = [{"role": "user", "content":prompt}]
    prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
    return prompt_with_chat_template

def get_question2(sample,tokenizer) : 
    prompt = f"""
Read the following conversation and generate a question to continue it, where the answer is one detail from the text:

Text: {sample['passage']}

Assistant: {sample['question1']}
User: {sample['answer1']}

Provide only the question, without any tags, comments, or references to the input text.
"""

    messages = [
        {"role": "system", "content": "You are an assistant who asks questions."},
        {"role": "user", "content": prompt}
    ]
    prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
    return prompt_with_chat_template

def get_question3(sample,tokenizer) : 
    prompt = f"""
Read the text below and create a specific question whose answer is one detail on the text and which is completely different form the previous question. 

text : {sample['passage']}
previous question : {sample['question2']}

Provide only the question, without any tags, comments, or references to the input text.
"""

    messages = [{"role": "user", "content":prompt}]
    prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
    return prompt_with_chat_template

def get_answer(sample, tokenizer, turn_number):
    # Construct the key for the question based on the turn number
    question_key = f"question{turn_number}"

    prompt = f"""
Answer the following question based on the provided text : 

text : {sample['passage']}
Question : {sample[question_key]}

Respond directly, avoiding any tags, comments, or references to the input text.
"""
    messages = [{"role": "user", "content": prompt}]
    prompt_with_chat_template = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return prompt_with_chat_template



def wiki(model_name: str, repo_name: str,output_path: str = './wiki_multi'):

    dataset = load_dataset("Slim205/top20_wiki_data")
    dataset = dataset['train'].select(range(50))
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
        prompts.append(get_question1(example,tokenizer))
    print(prompts[0])
    outputs = llm.generate(prompts,sampling_params)
    llm_question_1=[]
    for i, item in enumerate(outputs):
        output = item.outputs[0].text
        llm_question_1.append(output)
    dataset = dataset.add_column("question1",llm_question_1)
    prompts = []
    for example in dataset : 
        prompts.append(get_answer(example,tokenizer,1))

    print(prompts[0])

    outputs = llm.generate(prompts,sampling_params)
    llm_answers = []
    for i, item in enumerate(outputs):
        llm_answers.append(item.outputs[0].text)
    dataset = dataset.add_column("answer1", llm_answers)

    prompts = []
    for example in dataset:        
        prompts.append(get_question2(example,tokenizer))
    print(prompts[0])
    outputs = llm.generate(prompts,sampling_params)
    llm_question_1=[]
    for i, item in enumerate(outputs):
        output = item.outputs[0].text
        llm_question_1.append(output)
    dataset = dataset.add_column("question2",llm_question_1)

    prompts = []
    for example in dataset : 
        prompts.append(get_answer(example,tokenizer,2))

    print(prompts[0])

    outputs = llm.generate(prompts,sampling_params)
    llm_answers = []
    for i, item in enumerate(outputs):
        llm_answers.append(item.outputs[0].text)
    dataset = dataset.add_column("answer2", llm_answers)

    prompts = []
    for example in dataset:        
        prompts.append(get_question3(example,tokenizer))
    print(prompts[0])
    outputs = llm.generate(prompts,sampling_params)
    llm_question_1=[]
    for i, item in enumerate(outputs):
        output = item.outputs[0].text
        llm_question_1.append(output)
    dataset = dataset.add_column("question3",llm_question_1)

    prompts = []
    for example in dataset : 
        prompts.append(get_answer(example,tokenizer,3))

    print(prompts[0])

    outputs = llm.generate(prompts,sampling_params)
    llm_answers = []
    for i, item in enumerate(outputs):
        llm_answers.append(item.outputs[0].text)
    dataset = dataset.add_column("answer3", llm_answers)

    dataset_dict = DatasetDict({"train": dataset})

    #dataset_dict.save_to_disk(output_path)
    #print(f"Translated dataset saved to {output_path}")

    dataset_dict.push_to_hub(repo_name)
    print(f"Translated dataset saved and pushed to Hugging Face repo: {repo_name}")

if __name__ == '__main__':
    fire.Fire(wiki)