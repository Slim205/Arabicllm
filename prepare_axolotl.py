from datasets import load_dataset
import fire 

def create_conversation(example):
    return {
        "conversations": [
            {"from": "human", "value": example['translated_question']},
            {"from": "gpt", "value": example['translated_answer']}
        ]
    }

def data_filter(repo_name : str) :
  dataset = load_dataset(repo_name)
  processed_dataset = dataset.map(create_conversation ,remove_columns=dataset['train'].column_names)
  processed_dataset.push_to_hub(repo_name+'_ift')

if __name__ == '__main__':
    fire.Fire(data_filter)
