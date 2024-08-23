import fire
from vllm import LLM,SamplingParams
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer



def get_questions(text,tokenizer) : 
    prompt = f"""
Read the text below and formulate three distinct general questions, each of which requires lengthy and detailed answers drawn from different sections of the text. Separate the questions with a "$".

{text}

Provide only the questions, without any tags, comments, or references to the input text.
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
        prompts.append(get_questions(example['passage'],tokenizer))

    print(prompts[0])
    outputs = llm.generate(prompts,sampling_params)

    llm_question_1=[]
    llm_question_2=[]
    llm_question_3=[]
    list_passage=[]
    list_titles=[]
    list_links=[]
    list_index= []
    for i, item in enumerate(outputs):
        output = item.outputs[0].text
        if "$" in output : 
            list_questions = output.split('$')
            if len(list_questions) == 3 :
                llm_question_1.append(list_questions[0])
                llm_question_2.append(list_questions[1])
                llm_question_3.append(list_questions[2])
                list_passage.append(dataset['passage'][i])
                list_links.append(dataset['link'][i])
                list_titles.append(dataset['title'][i])
            else :
                list_index.append(i)
        else : 
            list_index.append(i)
    print('======================================================================\n')
    print(list_index)

    if list_index:
    # Filter out the problematic prompts
        prompts_to_redo = [prompts[i] for i in list_index]
        # Generate new outputs
        outputs_redo = llm.generate(prompts_to_redo, sampling_params)
        list_index_2 = []
        # Process the new outputs
        for i, item in enumerate(outputs_redo):
            output = item.outputs[0].text
            if "$" in output:
                list_questions = output.split('$')
                if len(list_questions) == 3:
                    idx = list_index[i]
                    llm_question_1.append(list_questions[0])
                    llm_question_2.append(list_questions[1])
                    llm_question_3.append(list_questions[2])
                    list_passage.append(dataset['passage'][idx])
                    list_links.append(dataset['link'][idx])
                    list_titles.append(dataset['title'][idx])
                else:
                    list_index_2.append(list_index[i])
            else:
                list_index_2.append(list_index[i])
    print('======================================================================\n')
    print(list_index_2)
    if list_index_2:
    # Filter out the problematic prompts
        prompts_to_redo = [prompts[i] for i in list_index_2]
        # Generate new outputs
        outputs_redo = llm.generate(prompts_to_redo, sampling_params)
        list_index_3 = []
        # Process the new outputs
        for i, item in enumerate(outputs_redo):
            output = item.outputs[0].text
            if "$" in output:
                list_questions = output.split('$')
                if len(list_questions) == 3:
                    idx = list_index_2[i]
                    llm_question_1.append(list_questions[0])
                    llm_question_2.append(list_questions[1])
                    llm_question_3.append(list_questions[2])
                    list_passage.append(dataset['passage'][idx])
                    list_links.append(dataset['link'][idx])
                    list_titles.append(dataset['title'][idx])
                else:
                    list_index_3.append(list_index_2[i])
            else:
                list_index_3.append(list_index_2[i])
    print('======================================================================\n')
    print(list_index_3)


    data = {'title' : list_titles, "passage" : list_passage, 'link':list_links,"question1": llm_question_1, "question2": llm_question_2, "question3" : llm_question_3}
    dataset = Dataset.from_dict(data)

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
    for example in dataset : 
        prompts.append(get_answer(example,tokenizer,2))
    print(prompts[0])
    outputs = llm.generate(prompts,sampling_params)
    llm_answers = []
    for i, item in enumerate(outputs):
        llm_answers.append(item.outputs[0].text)
    dataset = dataset.add_column("answer2", llm_answers)

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