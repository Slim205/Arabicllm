import fire 
from datasets import load_dataset

def verif(sample) : 
    return not (' text ' in sample['question']) and len(sample['answer'].split(' '))> 10 and len(sample['passage'].split(' ')) > 50

def create_conversation(example):
    return {
        "conversations": [
            {"from": "human", "value": example['translated_question']},
            {"from": "gpt", "value": example['translated_answer']}
        ]
    }

def data_filter() :
    repo_name = "Slim205/wiki_data_full"
    dataset = load_dataset(repo_name)
    dataset = dataset.filter(verif)
    print(len(dataset['train']))
    dataset.push_to_hub(repo_name+ '_filtered2')
    #processed_dataset = dataset.map(create_conversation ,remove_columns=dataset['train'].column_names)
    #processed_dataset.push_to_hub('Slim205/translated_wikipedia_10K_nllb')

if __name__ == '__main__':
    fire.Fire(data_filter)
