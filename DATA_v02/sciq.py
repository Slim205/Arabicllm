import fire
from vllm import LLM,SamplingParams
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer
import re 

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

def extract_generated_question(output: str) -> str:
    # Define the regex pattern to match the content between <generated_question> tags
    pattern = r"<generated_question>(.*?)</generated_question>"

    # Use re.search to find the match
    match = re.search(pattern, output, re.DOTALL)

    if match:
        # Return the content between the tags
        return match.group(1).strip()
    else:
        return None
def generate_chat_prompt2(sample, tokenizer):
    prompt = f"""
You are an AI assistant tasked with creating a multiple-choice question using an original question and a set of options. Follow these steps carefully:

1. Read the following question:
<question>
{sample['question']}
</question>

2. Now, consider the associated options:
<options>
    {sample['choice1']}  
    {sample['choice2']}  
    {sample['choice3']}  
    {sample['choice4']} 
</options>

3. Your task is to generate a multiple-choice question that incorporates all of the provided options, reflecting a choice between them. When creating your question:
   - Ensure that the new generated question reflects exactly the same sense as the old question
   - Ensure that all four options are included and relevant to the question
   - Keep the same order of the options as provided in the original set

4. Present your generated multiple-choice question in the following format:
   <generated_question>
   [Your question here]
   
   1. [Option 1]
   2. [Option 2]
   3. [Option 3]
   4. [Option 4]
   </generated_question>

Remember to number the options exactly as shown above (1, 2, 3, 4) and present each option on a separate line.
"""

    messages = [{'role': 'user', 'content': prompt}]
    prompt_with_chat_template = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True,  
        tokenize=False
    )
    
    return prompt_with_chat_template


def generate_chat_prompt(sample, tokenizer):
    prompt = f"""
You are an AI assistant tasked with creating a multiple-choice question using an incomplete sentence and a set of options. Follow these steps carefully:

1. Read the following incomplete_sentence:
<incomplete_sentence>
{sample['question']}
</incomplete_sentence>

2. Now, consider the associated options:
<options>
    {sample['choice1']}  
    {sample['choice2']}  
    {sample['choice3']}  
    {sample['choice4']} 
</options>

3. Your task is to generate a multiple-choice question that incorporates all of the provided options, reflecting a choice between them. When creating your question:
   - Identify the main subject and predicate of the statement.
   - Rearrange the words to form a question that asks about the information in the statement.
   - Ensure that all four options are included and relevant to the question.
   - Keep the same order of the options as provided in the original set.

4. Present your generated multiple-choice question in the following format:
   <generated_question>
   [Your question here]
   
   1. [Option 1]
   2. [Option 2]
   3. [Option 3]
   4. [Option 4]
   </generated_question>

Remember to number the options exactly as shown above (1, 2, 3, 4) and present each option on a separate line.
"""
    messages = [{'role': 'user', 'content': prompt}]
    prompt_with_chat_template = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True,  
        tokenize=False
    )
    
    return prompt_with_chat_template

def sciq(model_name: str, repo_name: str,output_path: str = './sciq'):
    dataset = load_dataset("allenai/sciq")
    dataset = dataset['train']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = transform_dataset(dataset)

    def get_answer(sample) : 
        prompt = f"""
Read the text below and answer the following question with a brief justification (one sentence) based on the provided support. Start your response with "{str(sample['label']+1)})" : 

Support: {sample['support']}
Question: {sample['question']}
Answer: {sample['correct_answer']}

Generate your response without including any tags, comments, or references to the provided text.
"""
        messages = [{"role": "user", "content":prompt}]
        prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
        return prompt_with_chat_template

    prompts = []
    for index,example in enumerate(dataset):
        if example['question'].strip(' ')[-1] != '?' :     
            prompts.append(generate_chat_prompt(example,tokenizer))
        else :
            prompts.append(generate_chat_prompt2(example,tokenizer))

    print(prompts[0])
    if model_name == "meta-llama/Meta-Llama-3.1-70B-Instruct" : 
        llm = LLM(model_name, tensor_parallel_size=4,max_model_len=2048,gpu_memory_utilization=0.93) 

    else :
        llm = LLM(model_name, max_model_len=4096) 
    sampling_params = SamplingParams(max_tokens=512,temperature=0.8, top_p=0.95)
    outputs = llm.generate(prompts,sampling_params)
    llm_questions=[]
    for i, item in enumerate(outputs):
        llm_questions.append(extract_generated_question(item.outputs[0].text))

    dataset = dataset.add_column("ift_question", llm_questions)

    llm_instructions = []
    for example in dataset : 
        llm_instructions.append(get_instruction(example['ift_question'],example['support']))
    dataset = dataset.add_column("ift_instruction", llm_instructions)

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
