import fire 
from datasets import load_dataset, concatenate_datasets

def create_conversation(example):
    return {
        "conversations": [
            {"from": "human", "value": example['translated_instruction']},
            {"from": "gpt", "value": example['translated_answer']}
        ]
    }

def data_filter(repo_name : str,multi_turn : bool = False) :
    dataset = load_dataset(repo_name)
    processed_dataset = dataset.map(create_conversation ,remove_columns=dataset['train'].column_names)
    if multi_turn :
        ds = load_dataset('Slim205/total_multi_to_train')
        ds_total = concatenate_datasets([processed_dataset['train'], ds['train']]).shuffle(seed=42)
        ds_total.push_to_hub(repo_name+'_ift')
    else : 
        processed_dataset.push_to_hub(repo_name+'_ift')

if __name__ == '__main__':
    fire.Fire(data_filter)
