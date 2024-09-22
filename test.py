from datasets import load_dataset, concatenate_datasets

list_data = [
    "Slim205/wiki_data_full_filtered_translated_nllb0",
    "Slim205/wiki_data_full_filtered_translated_nllb10000",
    "Slim205/wiki_data_full_filtered_translated_nllb20000",
    "Slim205/wiki_data_full_filtered_translated_nllb30000",
    "Slim205/wiki_data_full_filtered_translated_nllb40000",
    "Slim205/wiki_data_full_filtered_translated_nllb50000",
    "Slim205/wiki_data_full_filtered_translated_nllb60000",
    "Slim205/wiki_data_full_filtered_translated_nllb70000",
    "Slim205/wiki_data_full_filtered_translated_nllb80000",
    "Slim205/wiki_data_full_filtered_translated_nllb90000",
]

# Initialize ds_total as None to handle the first dataset
ds_total = None
data_exemple =  "Slim205/wiki_data_full_filtered"
for i in list_data:
    print(f"Loading dataset: {i}")
    ds = load_dataset(i)['train']
    if ds_total is None:
        ds_total = ds
    else:
        ds_total = concatenate_datasets([ds_total, ds])
# Push the concatenated dataset to the Hugging Face Hub
def verif(sample) : 
    return not (' text ' in sample['question']) and len(sample['answer'].split(' '))> 10 and len(sample['passage'].split(' ')) > 50
ds_total = ds_total.filter(verif)
ds_total.push_to_hub('Slim205/wiki_data_full_nllb')
