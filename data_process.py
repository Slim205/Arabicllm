from datasets import load_dataset
import re

def starts_with_english_word(sentence):
    pattern = r"^[a-zA-Z]+"
    match = re.match(pattern, sentence)
    return  not bool(match)

data = load_dataset("CohereForAI/aya_collection_language_split", "standard_arabic")

def filter_data(example):
    verif =  '<unk>' not in example['inputs'] and '<unk>' not in example['targets'] and example['script'] == 'Arab' and  starts_with_english_word(example['inputs']) and starts_with_english_word(example['targets'])
    return verif
filtered_dataset = data.filter(filter_data)

shuffled_dataset = filtered_dataset.shuffle(seed=42)

shuffled_dataset.push_to_hub("Slim205/aya_cleaned_v5_all_columns")