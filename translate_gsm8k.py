import fire
from vllm import LLM
from datasets import load_dataset, DatasetDict
import torch

def translate_gsm8k(model_name: str, language: str = 'Arabic', output_path: str = './translated_gsm8k.pt', repo_name: str = 'Slim205/translated-gsm8k'):

    # Load the dataset
    dataset = load_dataset("openai/gsm8k", "main")
    dataset = dataset['train'].select(range(10))
    
    # Initialize the LLM with the specified model
    llm = LLM(model_name,max_model_len=4096) #gpu_memory_utilization
    
    # Function to translate text
    def translate(text, target_language):
        prompt = f"Translate the following text to {target_language}:\n\n{text}"
        output = llm.generate(prompt)
        return output[0].outputs[0].text

    # Function to apply to each example
    def translate_example(example):
        example['question'] = translate(example['question'], language)
        example['answer'] = translate(example['answer'], language)
        return example

    # Apply translation to each example using map
    translated_dataset = dataset.map(translate_example)

    # Create a DatasetDict
    dataset_dict = DatasetDict({'train': translated_dataset})

    torch.save(dataset_dict, output_path)
    print(f"Translated dataset saved to {output_path}")

    # Push the dataset to Hugging Face
    dataset_dict.push_to_hub(repo_name)
    print(f"Translated dataset saved and pushed to Hugging Face repo: {repo_name}")

if __name__ == '__main__':
    fire.Fire(translate_gsm8k)
