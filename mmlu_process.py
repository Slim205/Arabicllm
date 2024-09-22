from datasets import load_from_disk
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the datasets
ds = load_from_disk("/gpfs/users/barkallasl/Arabicllm/DATA_v02/mmlu")

ds = ds['train']

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")

def remove_nans(example):
    return  not pd.isna(example['ift_answer'])

def remove_big_answer(sample) :
    return len(sample['ift_answer']) < 806 

def nllb(sample) : 
    return len(tokenizer.encode(sample['ift_answer']) ) < 256 and len(tokenizer.encode(sample['ift_instruction']) ) < 256

ds1 = ds.filter(remove_nans)
ds2 = ds1.filter(remove_big_answer)

#ds2.save_to_disk('mmlu_ift_filtered')
ds2.push_to_hub('Slim205/mmlu_ift_filtered')

ds3 = ds2.filter(nllb)
ds3.push_to_hub('Slim205/mmlu_256')
#ds3.save_to_disk('mmlu_256')