from datasets import load_dataset, Dataset, DatasetDict
import math
from datasets import concatenate_datasets
import json


def split_and_push_to_hub(dataset_name, split_size=10000):
    # Load the dataset
    dataset = load_dataset(dataset_name)
    dataset = dataset['train']
    print(dataset_name, len(dataset))
    if dataset_name == "Slim205/wiki_multi_full_filtered" : 
        split_size = 3000
    if dataset_name == "Slim205/race_ift" :
        split_size = 5000

    list_repo = []
    if len(dataset) > 15000 : 
    # Calculate the number of chunks
        num_chunks = math.ceil(len(dataset) / split_size)
        
        # Iterate over chunks and push each one to the Hub
        for i in range(num_chunks):
            start_idx = i * split_size
            end_idx = min((i + 1) * split_size, len(dataset))
            
            chunk = dataset.select(range(start_idx, end_idx))
            
            # Create a dataset dictionary
            chunk_dataset = DatasetDict({"train": chunk})

            # Push the chunk to the hub
            chunk_dataset.push_to_hub(f"{dataset_name}_chunk_{i+1}")

            print(f"Pushed chunk {i+1} to {dataset_name}_chunk_{i+1}")
            list_repo.append(f"{dataset_name}_chunk_{i+1}")
    else : 
        list_repo.append(dataset_name)

    return list_repo

if __name__ == '__main__':
    list_repo_name = ["Slim205/boolq_ift", "Slim205/race_ift","Slim205/copa_ift", "Slim205/hellaswag_ift",
                    "Slim205/sciq_ift", "Slim205/toxigen_ift", "Slim205/arc_challenge_ift", "Slim205/arc_easy_ift",
                    "Slim205/openbook_ift", "Slim205/piqa_ift",  "Slim205/gsm8k_ift",
                    "Slim205/wiki_data_full_filtered", "Slim205/wiki_multi_full_filtered"]

    all_repo_chunks = {}

    for repo in list_repo_name:
        repo_chunks = split_and_push_to_hub(repo)
        all_repo_chunks[repo] = repo_chunks

    list_new_data = []
    for x in all_repo_chunks.keys() : 
        list_new_data.extend(all_repo_chunks[x])

    print(list_new_data)
    # Save the dictionary to a JSON file
    with open('repo_chunks.json', 'w') as f:
        json.dump(all_repo_chunks, f)
