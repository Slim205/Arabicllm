def extract_arabic_translation(text):
    match = re.search(r"<translation>(.*?)</translation>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

async def translate_text(text):
    prompt = f"""
Here is the text to translate:

<text>
{text}
</text>

Please provide your translation now:
"""
    system_prompt = """You are tasked with translating an English text into Arabic. Your goal is to provide an accurate and natural-sounding translation.
    
Please follow these instructions carefully:

1. Translate the given text from English to Arabic.

2. If the text is a question:
   - Translate it as a question in Arabic
   - Do not provide an answer to the question

3. If the text contains both a statement and a question:
   - Translate both the statement and the question to Arabic
   - Keep the question as a question in Arabic
   - Do not provide an answer to the question

4. Present your translation inside <translation> tags.

5. If you encounter any culturally specific terms or idioms that don't have a direct Arabic equivalent, provide the best approximation.

6. Maintain the original punctuation style as much as possible in the Arabic translation.

7. If the original text uses quotation marks, use the appropriate Arabic quotation marks (« ») in your translation.
"""

    model_name = AvailableModel.BEDROCK_SONNET_3_5.value

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {"role": "user", "content": prompt},
    ]

    model = build_model(model_name, ModelParams(0.0, 4096))
    resp = await model.aquery(messages)

    return extract_arabic_translation(resp)


async def translate_sample(sample,data_repo_name):
    print(data_repo_name)
    if data_repo_name == "Slim205/wiki_multi_full_filtered" : 
        sample['translated_question1'] = await translate_text(sample['question1'])
        sample['translated_answer1'] = await translate_text(sample['answer1'])

        sample['translated_question2'] = await translate_text(sample['question2'])
        sample['translated_answer2'] = await translate_text(sample['answer2'])

        sample['translated_question3'] = await translate_text(sample['question3'])
        sample['translated_answer3'] = await translate_text(sample['answer3'])

    elif data_repo_name == "Slim205/wiki_data_full_filtered" :
        sample['translated_question'] = await translate_text(sample['question'])
        sample['translated_answer'] = await translate_text(sample['answer'])
    else :
        sample['translated_instruction'] = await translate_text(sample['ift_instruction'])
        sample['translated_answer'] = await translate_text(sample['ift_answer'])

    return sample

if __name__ == '__main__':
    import asyncio
    import os
    os.makedirs("/Users/pcolombo/Desktop/Slim205", exist_ok=True)
    async def main():
        
        top_10 = True
        list_repo_name =['Slim205/boolq_ift', 'Slim205/race_ift_chunk_1', 'Slim205/race_ift_chunk_2', 'Slim205/race_ift_chunk_3',
         'Slim205/race_ift_chunk_4', 'Slim205/race_ift_chunk_5', 'Slim205/race_ift_chunk_6', 'Slim205/race_ift_chunk_7',
          'Slim205/race_ift_chunk_8', 'Slim205/race_ift_chunk_9', 'Slim205/race_ift_chunk_10', 'Slim205/race_ift_chunk_11',
           'Slim205/race_ift_chunk_12', 'Slim205/race_ift_chunk_13', 'Slim205/race_ift_chunk_14', 'Slim205/race_ift_chunk_15',
            'Slim205/race_ift_chunk_16', 'Slim205/race_ift_chunk_17', 'Slim205/race_ift_chunk_18', 'Slim205/copa_ift', 
            'Slim205/hellaswag_ift_chunk_1', 'Slim205/hellaswag_ift_chunk_2', 'Slim205/hellaswag_ift_chunk_3', 
            'Slim205/hellaswag_ift_chunk_4', 'Slim205/sciq_ift', 'Slim205/toxigen_ift', 'Slim205/arc_challenge_ift',
             'Slim205/arc_easy_ift', 'Slim205/openbook_ift', 'Slim205/piqa_ift_chunk_1', 'Slim205/piqa_ift_chunk_2',
              'Slim205/gsm8k_ift', 'Slim205/wiki_data_full_filtered_chunk_1', 'Slim205/wiki_data_full_filtered_chunk_2',
 'Slim205/wiki_data_full_filtered_chunk_3', 'Slim205/wiki_data_full_filtered_chunk_4', 'Slim205/wiki_data_full_filtered_chunk_5',
  'Slim205/wiki_data_full_filtered_chunk_6', 'Slim205/wiki_data_full_filtered_chunk_7', 'Slim205/wiki_data_full_filtered_chunk_8',
   'Slim205/wiki_multi_full_filtered_chunk_1', 'Slim205/wiki_multi_full_filtered_chunk_2', 'Slim205/wiki_multi_full_filtered_chunk_3',
'Slim205/wiki_multi_full_filtered_chunk_4', 'Slim205/wiki_multi_full_filtered_chunk_5', 'Slim205/wiki_multi_full_filtered_chunk_6',
     'Slim205/wiki_multi_full_filtered_chunk_7', 'Slim205/wiki_multi_full_filtered_chunk_8']



        for data_repo_name in list_repo_name : 
            dataset = load_dataset(data_repo_name)
            print(data_repo_name)
            if top_10 : 
                dataset = dataset['train'].select(range(10))
            translated_dataset = []
            for sample in dataset:
                translated_sample = await translate_sample(sample, data_repo_name=data_repo_name)
                translated_dataset.append(translated_sample)
            translated_dataset = Dataset.from_list(translated_dataset)
            output_path = data_repo_name[8:]
            translated_dataset.save_to_disk(output_path)
            print(f"Translated dataset saved to {output_path}")
            
            #Define your hugging face repo : 
            translated_dataset.to_csv("/Users/pcolombo/Desktop/" + data_repo_name+'_translated.csv') #push_to_hub(data_repo_name+'_translated')
            print(f"Translated dataset saved and pushed to Hugging Face repo: {data_repo_name}_translated" )

    asyncio.run(main())





