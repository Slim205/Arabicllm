from datasets import load_dataset
#from transformers import AutoTokenizer

# Load the tokenizer
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Load the dataset
dataset = load_dataset("CohereForAI/aya_collection_language_split", "standard_arabic")
#dataset = load_dataset("Muennighoff/xP3x", "arb_Arab",trust_remote_code=True)

print('Data loaded')

def count_tokens(examples):
    tokenized_inputs = tokenizer(examples['inputs'])
    tokenized_outputs = tokenizer(examples['targets'])
    examples['num_input_tokens'] = [len(input_ids) for input_ids in tokenized_inputs['input_ids']]
    examples['num_output_tokens'] = [len(input_ids) for input_ids in tokenized_outputs['input_ids']]
    return examples

# Apply the function to the dataset with batched=True
tokenizer.pad_token = tokenizer.eos_token
tokenized_dataset = dataset.map(count_tokens, batched=True)

# List of splits to check
splits_to_check = ['train', 'test', 'validation']

# Summarize the token counts
total_input_tokens = 0
total_output_tokens = 0
split_tokens = {}

for split in splits_to_check:
    if split in tokenized_dataset:
        num_input_tokens = sum(tokenized_dataset[split]['num_input_tokens'])
        num_output_tokens = sum(tokenized_dataset[split]['num_output_tokens'])
        split_tokens[split] = {'input': num_input_tokens, 'output': num_output_tokens}
        print(f"Number of input tokens in {split} set: {num_input_tokens}")
        print(f"Number of output tokens in {split} set: {num_output_tokens}")
        total_input_tokens += num_input_tokens
        total_output_tokens += num_output_tokens

print('Total number of input tokens in the dataset:', total_input_tokens)
print('Total number of output tokens in the dataset:', total_output_tokens)

# Print total tokens per split
for split, tokens in split_tokens.items():
    print(f"{split.capitalize()} set - Input tokens: {tokens['input']}, Output tokens: {tokens['output']}")


