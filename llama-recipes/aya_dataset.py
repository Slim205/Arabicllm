import os
from datasets import load_dataset, DatasetDict
import torch

def get_custom_dataset(dataset_config, tokenizer, split, cache_dir="cached_dataset1",subset='first_half'):
    
   # split='train' #mooooooooooooodified 

    if os.path.exists(os.path.join(cache_dir, f"{split}_tokenized_dataset.pt")):
        print(f"Loading tokenized dataset from {cache_dir}")
        tokenized_dataset = torch.load(os.path.join(cache_dir, f"{split}_tokenized_dataset.pt"))

        if subset == "second_half":
            total_size = len(tokenized_dataset)
            start_index = total_size // 2
            tokenized_dataset = tokenized_dataset.select(range(start_index, total_size))
        elif subset == "first_half":
            total_size = len(tokenized_dataset)#*0+200
            end_index = total_size 
            if split == 'validation' :
                end_index = 100
            tokenized_dataset = tokenized_dataset.select(range(end_index))

        return tokenized_dataset


    dataset = load_dataset("CohereForAI/aya_collection_language_split", "standard_arabic")
    dataset = dataset['train'].select(range(4))  

    def tokenize_add_label(sample):
        messages = [
        {"role": "user", "content": sample['inputs']},
        {"role": "assistant", "content": sample['targets']}
        ]

        instruction = tokenizer.apply_chat_template(messages)
        answer = tokenizer.encode(sample["targets"] +  tokenizer.eos_token, add_special_tokens=False)
        
        sample = {
            "input_ids": instruction,
            "attention_mask" : [1] * len(instruction),
            "labels": [-100] * (len(instruction)-len(answer)) + answer,
            }
        return sample

    tokenized_dataset = dataset.map(tokenize_add_label, remove_columns=dataset.column_names)

  
    #torch.save(tokenized_dataset, os.path.join(cache_dir, f"{split}_tokenized_dataset.pt"))

    return tokenized_dataset
