import fire
from vllm import LLM,SamplingParams
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer
import re
def get_query_answer(line):
    text = line["text"]

    query = f"""
Does the following statement incite hatred?

'{text}'
    """
    if (line["toxicity_ai"] + line["toxicity_human"] ) > 5.5:
        answer = 'Yes'
    else :
        answer = 'No'
    return query,answer
    
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

def toxigen(model_name: str, repo_name: str, output_path: str = './toxigen'):
    dataset = load_dataset("toxigen/toxigen-data")
    dataset = dataset['train']#.select(range(10,30))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def get_question(sample) : 
        prompt = f"""
Your task is to generate a question that asks whether a given statement is hate speech or not. 

Here is the statement to be analyzed:
<statement>
{sample['text']}
</statement>

To generate the question:
1. Carefully read and understand the provided statement.
2. Formulate a question that asks whether this specific statement could be considered hate speech.
3. Ensure that your question is neutral and does not suggest an answer.
4. Include the entire statement verbatim in your question.

Present your generated question in the following format:
<generated_question>
[Your question here]
</generated_question>

Remember, your role is to generate a question, not to answer whether the statement is hate speech or not.
"""
        messages = [{"role": "user", "content":prompt}]
        prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
        return prompt_with_chat_template

    def get_answer(sample) : 
        prompt = f"""
Respond to this query with a brief justification (one sentence). Start your response with "{get_query_answer(sample)[1]}," : 

Query: {get_query_answer(sample)[0]}
Answer : {get_query_answer(sample)[1]}

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
    llm = LLM(model_name,max_model_len=4096) 
    sampling_params = SamplingParams(max_tokens=512,temperature=0.8, top_p=0.95)
    outputs = llm.generate(prompts,sampling_params)

    llm_answers=[]
    llm_questions=[]
    for i, item in enumerate(outputs):

        if i % 2 == 0 :
            llm_questions.append(extract_generated_question(item.outputs[0].text))
        else :
            llm_answers.append(item.outputs[0].text)

    dataset = dataset.add_column("ift_instruction", llm_questions)
    dataset = dataset.add_column("ift_answer", llm_answers)




    dataset_dict = DatasetDict({"train": dataset})

    dataset_dict.save_to_disk(output_path)
    print(f"Translated dataset saved to {output_path}")

    dataset_dict.push_to_hub(repo_name)
    print(f"Translated dataset saved and pushed to Hugging Face repo: {repo_name}")
if __name__ == '__main__':
    fire.Fire(toxigen)
