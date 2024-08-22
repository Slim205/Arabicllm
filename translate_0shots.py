import fire
from vllm import LLM,SamplingParams
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer

def translate_gsm8k():
    model_name = "google/gemma-2-27b-it"
    repo_name = "Slim205/wiki_translated_gemma"
    dataset = load_dataset("Slim205/wiki_data_more_2_filtered")
    dataset = dataset['train']#.select(range(100))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompts = []

    def apply_chat_template(text) : 
        prompt = f"""
Translate the following English text to Arabic :

« {text} »
        
Generate your response without including any tags, comments, or references to the provided text.
"""
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
        return prompt_with_chat_template


    for example in dataset:        
        prompts.append(apply_chat_template(example['question']))
        prompts.append(apply_chat_template(example['answer']))
    print(prompts[0])
    if model_name == "Qwen/Qwen2-72B-Instruct" : 
        llm = LLM(model_name, tensor_parallel_size=4,max_model_len=816,gpu_memory_utilization=0.95) 

    elif model_name == "google/gemma-2-27b-it" : 
        llm = LLM(model_name, tensor_parallel_size=2,max_model_len=2048,gpu_memory_utilization=0.85) 

    elif model_name == "meta-llama/Meta-Llama-3.1-70B-Instruct" : 
        llm = LLM(model_name, tensor_parallel_size=4,max_model_len=2048,gpu_memory_utilization=0.93) 

    else :
        llm = LLM(model_name,max_model_len=2048) 

    #sampling_params = SamplingParams(max_tokens=100,use_beam_search=True,early_stopping=True,best_of=3,temperature=0)
    sampling_params = SamplingParams(max_tokens=512,temperature=0.8, top_p=0.95)
    outputs = llm.generate(prompts,sampling_params)

    question_translated=[]
    answer_translated = []
    for i, item in enumerate(outputs):
        if i % 2 ==0 : 
            question_translated.append(item.outputs[0].text)
        else:
            answer_translated.append(item.outputs[0].text)

    data = { "question": question_translated, "answer": answer_translated}
    dataset = Dataset.from_dict(data)
    dataset_dict = DatasetDict({"train": dataset})

   # dataset_dict.save_to_disk(output_path)
   # print(f"Translated dataset saved to {output_path}")

    dataset_dict.push_to_hub(repo_name)
    print(f"Translated dataset saved and pushed to Hugging Face repo: {repo_name}")

if __name__ == '__main__':
    fire.Fire(translate_gsm8k)
