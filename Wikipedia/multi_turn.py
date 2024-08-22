from vllm import LLM,SamplingParams
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer
import fire


def create_conversation(example):
    return {
        "conversations": [
            {"from": "human", "value": example['question']},
            {"from": "gpt", "value": example['answer']},
            {"from": "human", "value": example['question_turn2']},
            {"from": "gpt", "value": example['answer_turn2']},
            {"from": "human", "value": example['question_turn3']},
            {"from": "gpt", "value": example['answer_turn3']},
            {"from": "human", "value": example['question_turn4']},
            {"from": "gpt", "value": example['answer_turn4']}
        ]
    }
def get_question(sample, tokenizer):
    prompt = f"""
Read the following conversation and generate a question to continue it.

Assistant: {sample['question']}
User: {sample['answer']}

Provide only the question, without any tags, comments, or references to the input text.
"""

    messages = [
        {"role": "system", "content": "You are an assistant who asks questions."},
        {"role": "user", "content": prompt}
    ]
    prompt_with_chat_template = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return prompt_with_chat_template

def get_question3(sample, tokenizer):
    prompt = f"""
Read the following conversation and generate a question to continue it.

Assistant: {sample['question']}
User: {sample['answer']}
Assistant: {sample['question_turn2']}
User: {sample['answer_turn2']}

Provide only the question, without any tags, comments, or references to the input text.
"""

    messages = [
        {"role": "system", "content": "You are an assistant who asks questions."},
        {"role": "user", "content": prompt}
    ]
    prompt_with_chat_template = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return prompt_with_chat_template

def get_question4(sample, tokenizer):
    prompt = f"""
Read the following conversation and generate a question to continue it.

Assistant: {sample['question']}
User: {sample['answer']}
Assistant: {sample['question_turn2']}
User: {sample['answer_turn2']}
Assistant: {sample['question_turn3']}
User: {sample['answer_turn3']}

Provide only the question, without any tags, comments, or references to the input text.
"""

    messages = [
        {"role": "system", "content": "You are an assistant who asks questions."},
        {"role": "user", "content": prompt}
    ]
    prompt_with_chat_template = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return prompt_with_chat_template


def get_answer(sample, tokenizer, turn_number):
    # Construct the key for the question based on the turn number
    question_key = f"question_turn{turn_number}"

    prompt = f"""
Answer the following question:

{sample[question_key]}

Respond directly, avoiding any tags, comments, or references to the input text.
"""
    messages = [{"role": "user", "content": prompt}]
    prompt_with_chat_template = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return prompt_with_chat_template

def multi_turn( model_name : str,repo_name : str):
    dataset = load_dataset("Slim205/wiki_data_test5_acva")
    dataset = dataset['train'].select(range(500))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_name == "google/gemma-2-27b-it" :
        llm = LLM(model_name,max_model_len=4096, tensor_parallel_size=2,gpu_memory_utilization=0.8)
    else :
        llm = LLM(model_name,max_model_len=4096) 
    sampling_params = SamplingParams(max_tokens=512,temperature=0.8, top_p=0.95)

    prompts = []

    for example in dataset : 
        prompts.append(get_question(example,tokenizer))

    print(prompts[0])

    outputs = llm.generate(prompts,sampling_params)
    llm_questions = []
    for i, item in enumerate(outputs):
        llm_questions.append(item.outputs[0].text)
    dataset = dataset.add_column("question_turn2", llm_questions)

    prompts = []
    for example in dataset : 
        prompts.append(get_answer(example,tokenizer,2))

    print(prompts[0])

    outputs = llm.generate(prompts,sampling_params)
    llm_answers = []
    for i, item in enumerate(outputs):
        llm_answers.append(item.outputs[0].text)
    dataset = dataset.add_column("answer_turn2", llm_answers)


    prompts = []

    for example in dataset : 
        prompts.append(get_question3(example,tokenizer))

    print(prompts[0])

    outputs = llm.generate(prompts,sampling_params)
    llm_questions = []
    for i, item in enumerate(outputs):
        llm_questions.append(item.outputs[0].text)
    dataset = dataset.add_column("question_turn3", llm_questions)

    prompts = []
    for example in dataset : 
        prompts.append(get_answer(example,tokenizer,3))

    print(prompts[0])

    outputs = llm.generate(prompts,sampling_params)
    llm_answers = []
    for i, item in enumerate(outputs):
        llm_answers.append(item.outputs[0].text)
    dataset = dataset.add_column("answer_turn3", llm_answers)
    prompts = []

    for example in dataset : 
        prompts.append(get_question4(example,tokenizer))

    print(prompts[0])

    outputs = llm.generate(prompts,sampling_params)
    llm_questions = []
    for i, item in enumerate(outputs):
        llm_questions.append(item.outputs[0].text)
    dataset = dataset.add_column("question_turn4", llm_questions)

    prompts = []
    for example in dataset : 
        prompts.append(get_answer(example,tokenizer,4))

    print(prompts[0])

    outputs = llm.generate(prompts,sampling_params)
    llm_answers = []
    for i, item in enumerate(outputs):
        llm_answers.append(item.outputs[0].text)
    dataset = dataset.add_column("answer_turn4", llm_answers)

#    dataset = dataset.map(create_conversation ,remove_columns=['question','answer','question_turn2','answer_turn2','question_turn3','answer_turn3','question_turn4','answer_turn4'])
    dataset_dict = DatasetDict({"train": dataset})

   # dataset_dict.save_to_disk(output_path)
   # print(f"Translated dataset saved to {output_path}")

    dataset_dict.push_to_hub(repo_name)
    print(f"Translated dataset saved and pushed to Hugging Face repo: {repo_name}")
 


if __name__ == '__main__':
    fire.Fire(multi_turn)
