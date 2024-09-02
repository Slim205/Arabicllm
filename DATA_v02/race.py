import fire
from vllm import LLM,SamplingParams
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer
import re

def get_correct_number(s) :
    if s=='A':
        return "1"
    elif  s=='B' :
        return "2"
    elif s == 'C' :
        return "3"
    else :
        return "4"

def get_instruction(question, text):
    instruction = f"""
    « {text} »

    {question}
    """
    return instruction

def get_correct_option(sample) :
    index = ord(sample['answer']) - ord('A')
    return sample['options'][index]

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

def extract_generated_answer(output: str) -> str:
    # Define the regex pattern to match the content between <generated_answer> tags
    pattern = r"<generated_answer>(.*?)</generated_answer>|<generated_answer>(.*)"
    
    # Use re.search to find the match
    match = re.search(pattern, output, re.DOTALL)
    
    if match:
        # Return the content between the tags, handling cases with or without the closing tag
        return (match.group(1) or match.group(2)).strip()
    else:
        # If no match found, return None
        return None

def generate_chat_prompt(sample, tokenizer):
    prompt = f"""
You are tasked with creating a multiple-choice question based on given context, a question, and a set of options. Follow these steps carefully:

1. Read the following context:
<context>
{sample['article']}
</context>

2. Now, consider this question and its associated options:
<question>
{sample['question']}
</question>

<options>
    {sample['options'][0]}  
    {sample['options'][1]}  
    {sample['options'][2]}  
    {sample['options'][3]} 
</options>

3. Your task is to generate a new multiple-choice question that incorporates all of the provided options, reflecting a choice between them.

4. When creating your question:
   - Ensure that all four options are included and relevant to the question.
   - Keep the same order of the options as provided in the original set.
   - If the original question is already a question, keep it as it is. 
   - If the original question contains '_' reformulate it to be a question.
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


def race(model_name: str, repo_name: str,output_path: str = './race'):
    dataset = load_dataset("ehovy/race","all")
    dataset = dataset['train'].select(range(50,60))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_answer(sample) : 
        prompt = f"""
You are an AI assistant tasked with reformulating a given answer to a question based on a provided text. Here's what you need to do:

First, carefully read the following text:
<support>
{sample['article']}
</support>

Now, consider this question:
<question>
{sample['ift_question']}
</question>

The correct answer to this question is:
<answer>
{ {get_correct_option(sample)}}
</answer>

Your task is to reformulate this answer by following these guidelines:
   a) Provide a brief justification (one sentence) for why this answer is correct.
   b) Even if you believe the answer might be incorrect based on the support text, defend it as if it were true.
   c) Do not mention any doubts about the answer's correctness.

Format your response as follows:
   <generated_answer>
   [Your one-sentence justification]
   </generated_answer>
"""

        messages = [{"role": "user", "content":prompt}]
        prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
        return prompt_with_chat_template

    prompts = []

    for example in dataset:        
        prompts.append(generate_chat_prompt(example,tokenizer))

    print(prompts[0])
    llm = LLM(model_name, max_model_len=4096) 
    sampling_params = SamplingParams(max_tokens=512,temperature=0.8, top_p=0.95)
    outputs = llm.generate(prompts,sampling_params)

    llm_instructions=[]
    llm_questions=[]
    for i, item in enumerate(outputs):
        llm_instructions.append(get_instruction(extract_generated_question(item.outputs[0].text),dataset[i]['article']))
        llm_questions.append(extract_generated_question(item.outputs[0].text))
    dataset = dataset.add_column("ift_instruction", llm_instructions)
    dataset = dataset.add_column("ift_question", llm_questions)

    prompts = []
    for example in dataset:        
        prompts.append(get_answer(example))
    print(prompts[0])
    outputs = llm.generate(prompts,sampling_params)
    llm_answers=[]
    for i, item in enumerate(outputs):
        output = extract_generated_answer(item.outputs[0].text)
        if output != None : 
            output = get_correct_number(dataset[i]['answer'])+') '+output

        llm_answers.append(output)  
    dataset = dataset.add_column("ift_answer", llm_answers)

    dataset_dict = DatasetDict({"train": dataset})

    dataset_dict.save_to_disk(output_path)
    print(f"Translated dataset saved to {output_path}")

    dataset_dict.push_to_hub(repo_name)
    print(f"Translated dataset saved and pushed to Hugging Face repo: {repo_name}")
if __name__ == '__main__':
    fire.Fire(race)
