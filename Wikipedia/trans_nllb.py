import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, DatasetDict,Dataset
import fire

def main(data_repo_name : str, batch_size : int = 4 ,start:int=0, end:int= 100000000) :
    model_name = "facebook/nllb-200-3.3B" #"facebook/nllb-200-distilled-600M"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda").eval()
  #  model = torch.compile(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def translate_text(text_list):
        inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True,max_length=2048).to("cuda")
        
        with torch.no_grad():
            translated_tokens = model.generate(**inputs, 
                                            forced_bos_token_id=tokenizer.convert_tokens_to_ids("arb_Arab"), 
                                            max_length=2048, 
                                            num_beams=5,
                                            )
        
        return [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]

    def translate_sample_batch(sample,data_repo_name):
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

    list_repo_name = ["Slim205/boolq_ift", "Slim205/race_ift", "Slim205/hellaswag_ift",
                   "Slim205/sciq_ift", "Slim205/toxigen_ift", 
                  "Slim205/openbook_ift", "Slim205/piqa_ift",  "Slim205/gsm8k_ift",
                "Slim205/wiki_multi_full_filtered"]

    dataset = load_dataset(data_repo_name)
    dataset = dataset['train'].select(range(start,min(end,len(dataset['train']))))
    translated_dataset = dataset.map(translate_sample_batch, batched=True, batch_size=batch_size,fn_kwargs={'data_repo_name': data_repo_name})
    output_path = data_repo_name[8:]
    translated_dataset.save_to_disk(output_path)
    print(f"Translated dataset saved to {output_path}")

    translated_dataset.push_to_hub(data_repo_name + '_translated_nllb'+str(start))

if __name__ == '__main__':
    fire.Fire(main)
