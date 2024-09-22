from datasets import load_dataset, concatenate_datasets

list_data = [
    "Slim205/sciq_ift_v02_filtered_translated_nllb10000",
    "Slim205/sciq_ift_v02_filtered_translated_nllb11000",
    "Slim205/sciq_ift_v02_filtered_translated_nllb10500",
        "Slim205/sciq_ift_v02_filtered_translated_nllb0",
    "Slim205/sciq_ift_v02_filtered_translated_nllb2000",
    "Slim205/sciq_ift_v02_filtered_translated_nllb4000",
    "Slim205/sciq_ift_v02_filtered_translated_nllb5000",
    "Slim205/sciq_ift_v02_filtered_translated_nllb6000",
    "Slim205/sciq_ift_v02_filtered_translated_nllb7000",
    "Slim205/sciq_ift_v02_filtered_translated_nllb8000",
    "Slim205/sciq_ift_v02_filtered_translated_nllb9000",
    "Slim205/arc_easy_ift_v02_filtered_translated_nllb0",
    "Slim205/arc_challenge_ift_v02_filtered_translated_nllb0",
    "Slim205/toxigen_ift_v02_filtered_translated_nllb0",
    "Slim205/toxigen_ift_v02_filtered_translated_nllb4000",

  "Slim205/copa_ift_v03_translated_nllb0" , #  
  #"Slim205/copa_ift_v02_filtered_translated_nllb0",

"Slim205/mmlu_nllb_256_filtered", #mmlu
"Slim205/race_256_nllb_filtered",      # race                                           
                                                  
    "Slim205/piqa_ift_v02_filtered_translated_nllb0",  
    "Slim205/wiki_data_full_nllb",
    "Slim205/openbook_ift_v02_filtered_translated_nllb0",
    "Slim205/openbook_ift_v02_filtered_translated_nllb2000",
   "Slim205/gsm8k_ift_v02_translated_nllb5000" , 
   "Slim205/gsm8k_ift_v02_translated_nllb0" , 
    "Slim205/hellaswag_ift_v02_filtered_translated_nllb10000",
    "Slim205/hellaswag_ift_v02_filtered_translated_nllb0",
    "Slim205/hellaswag_ift_v02_filtered_translated_nllb20000",
    "Slim205/hellaswag_ift_v02_filtered_translated_nllb24000",
    "Slim205/hellaswag_ift_v02_filtered_translated_nllb28000",
    "Slim205/hellaswag_ift_v02_filtered_translated_nllb32000",
    "Slim205/hellaswag_ift_v02_filtered_translated_nllb36000",
    "Slim205/boolq_ift_translated_nllb0",

"Slim205/acva_ift_translated_nllb0_mapped"
]

# Initialize ds_total as None to handle the first dataset
ds_total = None
data_exemple =  "Slim205/wiki_data_full_nllb"
for i in list_data:
    print(f"Loading dataset: {i}")
    ds = load_dataset(i)['train']
    if i == data_exemple : 
        ds = ds.rename_column('translated_question', 'translated_instruction')

    ds = ds.select_columns(['translated_instruction', 'translated_answer'])  # Select the relevant columns

    if ds_total is None:
        ds_total = ds
    else:
        ds_total = concatenate_datasets([ds_total, ds])
ds_total = ds_total.shuffle(seed=42)
# Push the concatenated dataset to the Hugging Face Hub

#This is the data without multi turn !!!

ds_total.push_to_hub('Slim205/total_data_to_train_acva')

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

data_filter('Slim205/total_data_to_train_acva',multi_turn=True)