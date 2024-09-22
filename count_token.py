from datasets import load_dataset
from transformers import AutoTokenizer

def compute_token_counts_for_repos(repo_list):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    # Function to count tokens
    def count_tokens(examples):
        
        tokenized_inputs = tokenizer(examples['question1'])
        tokenized_outputs = tokenizer(examples['answer1'])
        examples['num_input_tokens1'] = [len(input_ids) for input_ids in tokenized_inputs['input_ids']]
        examples['num_output_tokens1'] = [len(input_ids) for input_ids in tokenized_outputs['input_ids']]
        tokenized_inputs = tokenizer(examples['question2'])
        tokenized_outputs = tokenizer(examples['answer2'])
        examples['num_input_tokens2'] = [len(input_ids) for input_ids in tokenized_inputs['input_ids']]
        examples['num_output_tokens2'] = [len(input_ids) for input_ids in tokenized_outputs['input_ids']]
        tokenized_inputs = tokenizer(examples['question3'])
        tokenized_outputs = tokenizer(examples['answer3'])
        examples['num_input_tokens3'] = [len(input_ids) for input_ids in tokenized_inputs['input_ids']]
        examples['num_output_tokens3'] = [len(input_ids) for input_ids in tokenized_outputs['input_ids']]


        return examples


    # Iterate over each dataset in the repo list
    for repo_name in repo_list:
        print(f"\nProcessing dataset: {repo_name}")

        # Load the dataset
        dataset = load_dataset(repo_name)
        print('Data loaded')

        # Apply the function to the dataset with batched=True
        tokenizer.pad_token = tokenizer.eos_token
        tokenized_dataset = dataset.map(count_tokens, batched=True)

        # List of splits to check
        splits_to_check = ['train']

        # Summarize the token counts
        total_input_tokens = 0
        total_output_tokens = 0

        for i in range(1,4) : 
            num_input_tokens = sum(tokenized_dataset['train'][f'num_input_tokens{i}'])
            num_output_tokens = sum(tokenized_dataset['train'][f'num_output_tokens{i}'])
            total_input_tokens += num_input_tokens
            total_output_tokens += num_output_tokens

        print(f'Total number of input tokens in {repo_name}: {total_input_tokens}')
        print(f'Total number of output tokens in {repo_name}: {total_output_tokens}')

        # Print total tokens per split
        #for split, tokens in split_tokens.items():
         #   print(f"{split.capitalize()} set - Input tokens: {tokens['input']}, Output tokens: {tokens['output']}")

# List of dataset repositories
list_repo_name = [
    "Slim205/boolq_ift", "Slim205/race_ift_v02_filtered", "Slim205/copa_ift_v02_filtered", 
    "Slim205/hellaswag_ift_v02_filtered", "Slim205/sciq_ift_v02_filtered", 
    "Slim205/arc_challenge_ift_v02_filtered", "Slim205/arc_easy_ift_v02_filtered",
    "Slim205/piqa_ift_v02_filtered", "Slim205/gsm8k_ift_v02", 
    "Slim205/wiki_data_full_filtered", "Slim205/multi_turn_v02_filtered"
]

# Run the function for all repos
compute_token_counts_for_repos(["Slim205/multi_turn_v02_filtered"])
