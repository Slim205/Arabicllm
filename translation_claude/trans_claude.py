from datasets import load_dataset, DatasetDict,Dataset
import anthropic
import fire

def translate_text(text):

    client = anthropic.Anthropic(api_key="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")
    prompt = f"""
Translate the following English text to Arabic :

« {text} »
        
Generate your response without including any tags, comments, or references to the provided text.
"""
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=2048,
        system="You are an assistant who translates English texts into Arabic.",
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
    return message.content[0].text


def translate_sample(sample,data_repo_name):
    print(data_repo_name)
    if data_repo_name == "Slim205/wiki_multi_full" : 
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
        translated_dataset.push_to_hub(data_repo_name+'_translated')
        print(f"Translated dataset saved and pushed to Hugging Face repo: {data_repo_name}_translated" )

if __name__ == '__main__':
    fire.Fire(main)
