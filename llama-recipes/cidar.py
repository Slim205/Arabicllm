import os
from datasets import load_dataset, DatasetDict

def get_custom_dataset(dataset_config, tokenizer, split):

    dataset = load_dataset("arbml/CIDAR")
    if split == 'train' : 

        dataset = dataset['train'].select(range(9900))
    else :
        dataset = dataset['train'].select(range(9900,10000))


    def tokenize_add_label(sample):
        messages = [
        {"role": "user", "content": sample['instruction']},
        {"role": "assistant", "content": sample['output']}
        ]

        instruction = tokenizer.apply_chat_template(messages)
        answer = tokenizer.encode(sample["output"] +  tokenizer.eos_token, add_special_tokens=False)
        
        sample = {
            "input_ids": instruction,
            "attention_mask" : [1] * len(instruction),
            "labels": [-100] * (len(instruction)-len(answer)) + answer,
            }
        return sample

    tokenized_dataset = dataset.map(tokenize_add_label, remove_columns=dataset.column_names)

    return tokenized_dataset
