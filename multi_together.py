from datasets import load_dataset, concatenate_datasets

list_data = [
"Slim205/wiki_multi_full_filtered_translated_nllb0",
"Slim205/wiki_multi_full_filtered_translated_nllb3000",
"Slim205/wiki_multi_full_filtered_translated_nllb6000",
"Slim205/wiki_multi_full_filtered_translated_nllb9000",
"Slim205/wiki_multi_full_filtered_translated_nllb12000",
"Slim205/wiki_multi_full_filtered_translated_nllb15000",
"Slim205/wiki_multi_full_filtered_translated_nllb18000",
"Slim205/wiki_multi_full_filtered_translated_nllb21000"

]
def create_conversation(example):
    return {
        "conversations": [
            {"from": "human", "value": example['translated_question1']},
            {"from": "gpt", "value": example['translated_answer1']},
            {"from": "human", "value": example['translated_question2']},
            {"from": "gpt", "value": example['translated_answer2']},
            {"from": "human", "value": example['translated_question3']},
            {"from": "gpt", "value": example['translated_answer3']}

        ]
    }
def filter_data(sample) : 
    return  not (' text ' in sample['answer1']) and  not (' text.' in sample['answer1']) and  not (' text ' in sample['answer2']) and  not (' text.' in sample['answer2']) and  not (' text ' in sample['answer3']) and  not (' text.' in sample['answer3'])

ds_total = None
for i in list_data:
    print(f"Loading dataset: {i}")
    ds = load_dataset(i)['train']
    if ds_total is None:
        ds_total = ds
    else:
        ds_total = concatenate_datasets([ds_total, ds])
ds_total = ds_total.filter(filter_data)
processed_dataset = ds_total.map(create_conversation ,remove_columns=ds_total.column_names)

#ds_total = ds_total.shuffle(seed=42)
# Push the concatenated dataset to the Hugging Face Hub
ds_total.push_to_hub('Slim205/multi_turn_nllb')
