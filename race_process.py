from datasets import load_dataset, concatenate_datasets
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the datasets
ds = load_dataset("Slim205/race_ift_v02_filtered")

ds = ds['train']

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")

def nllb(sample) : 
    return len(tokenizer.encode(sample['ift_answer']) ) < 256 and len(tokenizer.encode(sample['ift_instruction']) ) < 256

ds3 = ds.filter(nllb)

ds3.push_to_hub('Slim205/race_256')