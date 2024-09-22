import random
from datasets import load_dataset, concatenate_datasets

def fewshot(ds):
    ds = ds.shuffle()  
    index = len(ds['train']) // 4 
    ds_few_shot = ds['train'].select(range(index))  # 25% of the data
    ds_rest = ds['train'].select(range(index, len(ds['train'])))


    def get_shots(sample):
        num_shots = random.choice([1, 3, 5])
        new_instruction = ""
        ds_rest_shuffled = ds_rest.shuffle()  # Set a seed for reproducibility
        samples = ds_rest_shuffled.select(range(num_shots))
        for i in range(num_shots):
            new_instruction += samples[i]['translated_instruction'] + '\n' + samples[i]['translated_answer'] + '\n' + '\n'
        new_instruction += sample['translated_instruction']
        
        sample['new_instruction'] = new_instruction
        return sample  # Return the sample with the new field
    def get_0_shots(sample) :
        sample['new_instruction'] = sample['translated_instruction']
        return sample
    ds_few_shot = ds_few_shot.map(get_shots)
    ds_rest = ds_rest.map(get_0_shots)
    ds_total = concatenate_datasets([ds_few_shot, ds_rest])
    ds_total = ds_total.shuffle()  # Set a seed for reproducibility
    return ds_total
if __name__ == '__main__':
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
]
    ds_total = None
    for i in list_data:
        print(f"Loading dataset: {i}")
        ds2 = load_dataset(i)
        ds1 = fewshot(ds2)
        ds1 = ds1.select_columns(['new_instruction', 'translated_answer'])  

        if ds_total is None:
            ds_total = ds1
        else:
            ds_total = concatenate_datasets([ds_total, ds1])

    ds_total.push_to_hub('Slim205/data_fewshots')
    
    ds = load_dataset( "Slim205/wiki_data_full_nllb")
    ds = ds.rename_column('translated_question', 'translated_instruction')
    ds = ds.select_columns(['translated_instruction', 'translated_answer'])  

    ds_total = ds_total.rename_column('new_instruction', 'translated_instruction')
    ds_total = concatenate_datasets([ds_total, ds['train']])

    ds_total = ds_total.shuffle()

    ds_total.push_to_hub('Slim205/data_fewshots')

    