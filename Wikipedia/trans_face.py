import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, DatasetDict,Dataset
import fire

def main(data_repo_name : str, batch_size : int = 4 ,start:int=0, end:int= 100000000) :
    model_name = "facebook/seamless-m4t-v2-large" #"facebook/nllb-200-distilled-600M"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda").eval()
  #  model = torch.compile(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def translate_text(text_list):
        inputs = tokenizer(text_list, return_tensors="pt", padding=True,truncation=True,max_length=8152).to("cuda")
        
        with torch.no_grad():
            translated_tokens = model.generate(**inputs, 
                                             tgt_lang="arb",
                                            max_new_tokens=4096, 
                                            #num_beams=5,
                                            do_sample = True,
                                            top_p =.95
                                            )
        
        return [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]

    def translate_sample_batch(sample,data_repo_name):
        sample['translated_instruction'] = translate_text(sample['ift_instruction'])
        sample['translated_answer'] = translate_text(sample['ift_answer'])

        return sample

    dataset = load_dataset(data_repo_name)
    dataset = dataset['train'].select(range(start,min(end,len(dataset['train']))))
    translated_dataset = dataset.map(translate_sample_batch, batched=True, batch_size=batch_size,fn_kwargs={'data_repo_name': data_repo_name})
    output_path = data_repo_name[8:]
    translated_dataset.save_to_disk(output_path)
    print(f"Translated dataset saved to {output_path}")

    translated_dataset.push_to_hub(data_repo_name + '_translated_seamless'+str(start))

if __name__ == '__main__':
    fire.Fire(main)
