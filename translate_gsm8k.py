import fire
from vllm import LLM
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer

def translate_gsm8k(model_name: str, target_language: str = 'Arabic', output_path: str = './translated_gsm8k', repo_name: str = 'Slim205/translated-gsm8k'):

    dataset = load_dataset("openai/gsm8k", "main")
    dataset = dataset['train'].select(range(10))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompts = []

    def apply_chat_template(text) : 
        prompt = [{"role": "user", "content": f"Translate the following text to {target_language}:\n\n{text}"},{"role": "assistant", "content" : "" }]
        inputs= tokenizer.apply_chat_template(prompt)
        return tokenizer.decode(inputs[:-1])

    for example in dataset:        
        prompts.append(apply_chat_template(example['question']))
        prompts.append(apply_chat_template(example['answer']))

    llm = LLM(model_name)
    outputs = llm.generate(prompts)

    question_translated=[]
    answer_translated = []
    for i, item in enumerate(outputs):
        if i % 2 ==0 : 
            question_translated.append(item.outputs[0].text)
        else:
            answer_translated.append(item.outputs[0].text)

    data = { "question": question_translated, "answer": answer_translated}
    dataset = Dataset.from_dict(data)
    dataset_dict = DatasetDict({"train": dataset})

    dataset_dict.save_to_disk(output_path)
    print(f"Translated dataset saved to {output_path}")

    dataset_dict.push_to_hub(repo_name)
    print(f"Translated dataset saved and pushed to Hugging Face repo: {repo_name}")

if __name__ == '__main__':
    fire.Fire(translate_gsm8k)
