import fire
from vllm import LLM,SamplingParams
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer
import re

def get_str(s) :
    if int(s)==0:
        return "first"
    elif  int(s)==1 :
        return "second"
    elif int(s) == 2 :
        return "third"
    else :
        return "last"

def get_instruction(question, text):
    instruction = f"""
    « {text} »

    {question}
    """
    return instruction
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

def generate_chat_prompt(sample, tokenizer):
    prompt = f"""
You are an AI assistant tasked with creating a multiple-choice question based on given context and a set of options. Follow these steps carefully:

1. Read the following context:
<context>
{sample['ctx']}
</context>

2. Now, consider the associated options for endings:
<options>
    {sample['endings'][0]} 
    {sample['endings'][1]}
    {sample['endings'][2]}
    {sample['endings'][3]}
</options>

3. Your task is to generate a multiple-choice question that incorporates all of the provided options, reflecting a choice between them. The options are possible endings to complete the context.

4. When creating your question:
   - Ensure that all four options are included and relevant to the question.
   - Keep the same order of the options as provided in the original set.
   - The question should be clear and unambiguous.

5. Present your generated multiple-choice question in the following format:
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

def hellaswag(model_name: str, repo_name: str,output_path: str = './hellaswag'):

    dataset = load_dataset("Rowan/hellaswag")
    dataset = dataset['train']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_question(sample) : 
        system_prompt = "Create a question that incorporates the following context reflecting a choice between the options. Include all options in the question in the same order as given"
        prompt = f"""
Options: {sample['endings'][0]}, {sample['endings'][1]}, {sample['endings'][2]}, {sample['endings'][3]}
Context: {sample['ctx']}

Generate the question directly, without adding any tags, comments, or references to the input text.
"""

        messages = [{'role':'system', 'content' : system_prompt},{"role": "user", "content":prompt}]
        prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
        return prompt_with_chat_template

    def get_correct_option(sample) :
        index = int(sample['label'])
        return sample['endings'][index]
    def get_answer(sample) : 
        prompt = f"""
Answer the following question with a brief justification (one sentence). Start your response with "{str(int(sample['label'])+1)})" : 

{sample['ift_instruction']}
Answer : {get_correct_option(sample)}

Respond directly, avoiding any tags, comments, or references to the input text.
"""

        messages = [{"role": "user", "content":prompt}]
        prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
        return prompt_with_chat_template

    prompts = []

    for example in dataset:        
        prompts.append(generate_chat_prompt(example,tokenizer))

    print(prompts[0])
    if model_name == "meta-llama/Meta-Llama-3.1-70B-Instruct" : 
        llm = LLM(model_name, tensor_parallel_size=4,max_model_len=2048,gpu_memory_utilization=0.93) 

    else :
        llm = LLM(model_name, max_model_len=4096,tensor_parallel_size=4) 
    sampling_params = SamplingParams(max_tokens=512,temperature=0.8, top_p=0.95)
    outputs = llm.generate(prompts,sampling_params)
    llm_questions=[]
    llm_instructions=[]

    for i, item in enumerate(outputs):
        output = item.outputs[0].text
        
        llm_instructions.append(get_instruction(extract_generated_question(output), dataset['ctx_a'][i]))
        llm_questions.append(extract_generated_question(output))
    dataset = dataset.add_column("ift_instruction", llm_instructions)
    dataset = dataset.add_column("ift_question", llm_questions)


    prompts = []

    for example in dataset:        
        prompts.append(get_answer(example))

    print(prompts[0])
    outputs = llm.generate(prompts,sampling_params)
    
    llm_answers=[]
    for i, item in enumerate(outputs):
        llm_answers.append(item.outputs[0].text)

    dataset = dataset.add_column("ift_answer", llm_answers)

    dataset_dict = DatasetDict({"train": dataset})

    dataset_dict.save_to_disk(output_path)
    print(f"Translated dataset saved to {output_path}")

    dataset_dict.push_to_hub(repo_name)
    print(f"Translated dataset saved and pushed to Hugging Face repo: {repo_name}")
if __name__ == '__main__':
    fire.Fire(hellaswag)
