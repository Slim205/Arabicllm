import fire
from vllm import LLM,SamplingParams
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer
import torch
from datasets import Dataset

def transform_dataset(data):
    def transform_example(example):
        # Create a list of choices by including both wrong and correct answers
        choices = [example['distractor3'], example['distractor1'], example['distractor2'], example['correct_answer']]
        
        # Shuffle the choices (optional) to randomize their order
        import random
        random.shuffle(choices)
        
        # Find the index of the correct answer
        label = choices.index(example['correct_answer'])
        
        return {
            'question': example['question'],
            'choice1': choices[0],
            'choice2': choices[1],
            'choice3': choices[2],
            'choice4': choices[3],
            'label': label
        }
    
    # Apply the transformation to the dataset
    transformed_data = data.map(transform_example)
    return transformed_data
def get_str(s) :
    if s==0:
        return "first"
    elif  s==1 :
        return "second"
    elif s == 2 :
        return "third"
    else :
        return "last"

def get_instruction(question, text):
    instruction = f"""
    « {text} »

    {question}
    """
    return instruction

def get_correct_option(sample) :
    column_name = 'choice' + str(sample['label']+1)
    return sample[column_name]

def sciq(model_name: str, repo_name: str,output_path: str = './sciq'):
    dataset = load_dataset("allenai/sciq")
    dataset = dataset['train']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = transform_dataset(dataset)

    def get_question(sample) : 
        prompt = f"""
Please create a more detailed question that incorporates the following options, reflecting a choice between them. Include all options in the question.

Options: {sample['choice1']}, {sample['choice2']}, {sample['choice3']}, {sample['choice4']}
Question: {sample['question']}

Generate the question directly, without adding any tags, comments, or references to the input text.
"""

        messages = [{"role": "user", "content":prompt}]
        prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
        return prompt_with_chat_template

    def get_answer(sample) : 
        prompt = f"""
Read the text below and answer the following question with a brief justification (one sentence). Start your response with "The {get_str(sample['label'])} option is correct." : 

Text: {sample['support']}
Question: {sample['question']}
Answer: {sample['correct_answer']}

Generate your response without including any tags, comments, or references to the provided text.
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
        llm_questions.append(get_instruction(item.outputs[0].text,dataset[i]['support']))
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

    columns_to_remove = [ "choice1", "choice2", "choice3", "choice4", "label"]
    dataset = dataset.remove_columns(columns_to_remove)


    dataset_dict = DatasetDict({"train": dataset})

    dataset_dict.save_to_disk(output_path)
    print(f"Translated dataset saved to {output_path}")

    dataset_dict.push_to_hub(repo_name)
    print(f"Translated dataset saved and pushed to Hugging Face repo: {repo_name}")
if __name__ == '__main__':
    fire.Fire(sciq)
