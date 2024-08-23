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
        max_tokens=1000,
        system="You are an assistant who translate English texts into Arabic.",
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
    sample['translated_instruction'] = translate_text(sample['ift_instruction'])
    sample['translated_answer'] = translate_text(sample['ift_answer'])
    return sample


def translate( data_repo_name: str):

    dataset = load_dataset(data_repo_name)
    translated_dataset = dataset.map(translate_sample)

    output_path = data_repo_name[8:]
    translated_dataset.save_to_disk(output_path)
    print(f"Translated dataset saved to {output_path}")

    translated_dataset.push_to_hub(data_repo_name+'_translated')
    print(f"Translated dataset saved and pushed to Hugging Face repo: {data_repo_name}_translated" )

if __name__ == '__main__':
    fire.Fire(translate)
