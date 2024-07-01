from datasets import load_dataset
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Load the dataset
dataset = load_dataset("CohereForAI/aya_collection_language_split", "standard_arabic")
#dataset = load_dataset("Muennighoff/xP3x", "arb_Arab",trust_remote_code=True)
print('Data loaded')

def count_tokens(examples):
    inp_out = [inp + out for inp, out in zip(examples['inputs'], examples['targets'])]
    tokenized_outputs = tokenizer(inp_out, truncation=True, padding=True)
    examples['num_tokens'] = [len(input_ids) for input_ids in tokenized_outputs['input_ids']]
    return examples

# Apply the function to the dataset with batched=True
tokenizer.pad_token = tokenizer.eos_token
tokenized_dataset = dataset.map(count_tokens, batched=True)

# List of splits to check
splits_to_check = ['train', 'test', 'validation']

# Summarize the token counts
total_tokens = 0
for split in splits_to_check:
    if split in tokenized_dataset:
        num_tokens = sum(tokenized_dataset[split]['num_tokens'])
        print(f"Number of tokens in {split} set: {num_tokens}")
        total_tokens += num_tokens

print('Total number of tokens in the dataset:', total_tokens)
