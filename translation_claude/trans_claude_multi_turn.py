from datasets import load_dataset, DatasetDict,Dataset
import anthropic
import fire

def translate_text(text):

    client = anthropic.Anthropic()
    prompt = f"""
Translate the following English text to Arabic :

« {text} »
        
Generate your response without including any tags, comments, or references to the provided text.
"""
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=512,
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


def translate_sample(sample):
    sample['translated_question1'] = translate_text(sample['question1'])
    sample['translated_answer1'] = translate_text(sample['answer1'])

    sample['translated_question2'] = translate_text(sample['question2'])
    sample['translated_answer2'] = translate_text(sample['answer2'])

    sample['translated_question3'] = translate_text(sample['question3'])
    sample['translated_answer3'] = translate_text(sample['answer3'])
    return sample

def main():
    data_repo_name = "Slim205/wiki_multi_full"
    dataset = load_dataset(data_repo_name)

    translated_dataset = dataset.map(translate_sample)

    output_path = "wiki_multi_full"
    translated_dataset.save_to_disk(output_path)
    print(f"Translated dataset saved to {output_path}")

    translated_dataset.push_to_hub(data_repo_name+'_translated')
    print(f"Translated dataset saved and pushed to Hugging Face repo: {data_repo_name}_translated" )

if __name__ == '__main__':
    main()