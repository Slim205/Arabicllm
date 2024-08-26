import time
from datasets import load_dataset, DatasetDict, Dataset
import anthropic
import fire

def translate_text(text):
    start_time = time.time()

    client = anthropic.Anthropic()
    prompt = f"""
Translate the following English text to Arabic :

« {text} »
        
Generate your response without including any tags, comments, or references to the provided text.
"""
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
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

    end_time = time.time()
    print(f"Translation time for one text: {end_time - start_time} seconds")
    return message.content[0].text


def translate_sample(sample):
    start_time = time.time()

    sample['translated_instruction'] = translate_text(sample['ift_instruction'])
    sample['translated_answer'] = translate_text(sample['ift_answer'])

    end_time = time.time()
    print(f"Translation time for one sample: {end_time - start_time} seconds")
    return sample


def translate(data_repo_name: str):
    start_time = time.time()

    dataset = load_dataset(data_repo_name)
    print(f"Dataset loading time: {time.time() - start_time} seconds")

    start_time = time.time()
    translated_dataset = dataset.map(translate_sample)
    print(f"Total translation time: {time.time() - start_time} seconds")

    output_path = data_repo_name[8:]

    start_time = time.time()
    translated_dataset.save_to_disk(output_path)
    print(f"Dataset saving time: {time.time() - start_time} seconds")
    
    start_time = time.time()
    translated_dataset.push_to_hub(data_repo_name + '_translated')
    print(f"Pushing to Hugging Face Hub time: {time.time() - start_time} seconds")

    print(f"Translated dataset saved to {output_path} and pushed to Hugging Face repo: {data_repo_name}_translated")


if __name__ == '__main__':
    fire.Fire(translate)
