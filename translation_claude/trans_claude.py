from datasets import load_dataset, DatasetDict,Dataset
import anthropic
import fire
import re

def extract_arabic_translation(text):
    match = re.search(r"<translation>(.*?)</translation>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def translate_text(text):

    client = anthropic.Anthropic(api_key="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")
    prompt = f"""
Here is the text to translate:

<text>
{text}
</text>

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

5. If you encounter any culturally specific terms or idioms that don't have a direct Arabic equivalent, provide the best approximation and include a brief explanation in parentheses after the term.

6. Maintain the original punctuation style as much as possible in the Arabic translation.

7. If the original text uses quotation marks, use the appropriate Arabic quotation marks (« ») in your translation.

Please provide your translation now:
"""
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=8192,
        system="You are tasked with translating an English text into Arabic. Your goal is to provide an accurate and natural-sounding translation.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    return extract_arabic_translation(message.content[0].text)


def translate_sample(sample,data_repo_name):
    print(data_repo_name)
    if data_repo_name == "Slim205/wiki_multi_full_filtered" : 
        sample['translated_question1'] = translate_text(sample['question1'])
        sample['translated_answer1'] = translate_text(sample['answer1'])

        sample['translated_question2'] = translate_text(sample['question2'])
        sample['translated_answer2'] = translate_text(sample['answer2'])

        sample['translated_question3'] = translate_text(sample['question3'])
        sample['translated_answer3'] = translate_text(sample['answer3'])

    elif data_repo_name == "Slim205/wiki_data_full_filtered" :
        sample['translated_question'] = translate_text(sample['question'])
        sample['translated_answer'] = translate_text(sample['answer'])
    else :
        sample['translated_instruction'] = translate_text(sample['ift_instruction'])
        sample['translated_answer'] = translate_text(sample['ift_answer'])

    return sample

def main(top_10 : bool):
    list_repo_name = ["Slim205/wiki_data_full_filtered", "Slim205/wiki_multi_full", "Slim205/boolq_ift", "Slim205/race_ift",
                  "Slim205/copa_ift", "Slim205/hellaswag_ift", "Slim205/sciq_ift", "Slim205/toxigen_ift",
                  "Slim205/arc_challenge_ift", "Slim205/arc_easy_ift", "Slim205/openbook_ift", "Slim205/piqa_ift",
                  "Slim205/gsm8k_ift"]
    for data_repo_name in list_repo_name : 
        dataset = load_dataset(data_repo_name)
        print(data_repo_name)
        if top_10 : 
            dataset = dataset['train'].select(range(10))
        translated_dataset = dataset.map(translate_sample,fn_kwargs={'data_repo_name': data_repo_name})
        output_path = data_repo_name[8:]
        translated_dataset.save_to_disk(output_path)
        print(f"Translated dataset saved to {output_path}")
        
        #Define your hugging face repo : 
        translated_dataset.push_to_hub(data_repo_name+'_translated_4')
        print(f"Translated dataset saved and pushed to Hugging Face repo: {data_repo_name}_translated" )

if __name__ == '__main__':
    fire.Fire(main)
