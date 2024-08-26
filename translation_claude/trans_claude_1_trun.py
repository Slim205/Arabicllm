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
    prompt0 = f"""
You are tasked with translating an English text into Arabic. Your goal is to provide an accurate and natural-sounding translation that preserves the meaning and tone of the original text.

Here is the English text to be translated:

<english_text>
{text}
</english_text>

Please follow these steps:

1. Read the English text carefully to understand its meaning and context.
2. Translate the text into Arabic, ensuring that you maintain the original meaning, tone, and style as closely as possible.
3. Pay attention to cultural nuances and idiomatic expressions, adapting them appropriately for an Arabic-speaking audience when necessary.
4. Review your translation to ensure it flows naturally in Arabic and accurately conveys the intended message.

Provide your Arabic translation without any additional comments, explanations, or references to the original text. Do not include any XML tags or other formatting in your response. Simply output the Arabic translation as plain text.
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
                        "text": prompt0
                    }
                ]
            }
        ]
    )
    return message.content[0].text


def translate_sample(sample):
        
    sample['translated_question'] = translate_text(sample['question'])
    sample['translated_answer'] = translate_text(sample['answer'])
    
    return sample

def main():
    data_repo_name = "Slim205/wiki_data_full_filtered"
    dataset = load_dataset(data_repo_name)
    dataset = dataset['train'].select(range(50))
    translated_dataset = dataset.map(translate_sample)
    
    output_path = "wiki_data_full"
    translated_dataset.save_to_disk(output_path)
    print(f"Translated dataset saved to {output_path}")

    translated_dataset.push_to_hub(data_repo_name+'_translated')
    print(f"Translated dataset saved and pushed to Hugging Face repo: {data_repo_name}_translated" )

if __name__ == '__main__':
    main()
