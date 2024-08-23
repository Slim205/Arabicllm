import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset

model_name = "facebook/nllb-200-3.3B"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda").eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

def translate_batch(text_list):
    inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True).to("cuda")
    translated_tokens = model.generate(**inputs, 
                                       forced_bos_token_id=tokenizer.convert_tokens_to_ids("arb_Arab"), 
                                       max_length=512, 
                                        num_beams=5,
                                        early_stopping=True
                                    )
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]

def translate_sample_batch(samples):
    questions = samples['question']
    answers = samples['answer']
    
    translated_questions = translate_batch(questions)
    translated_answers = translate_batch(answers)
    
    return {
        'question': questions,
        'answer': answers,
        'translated_question': translated_questions,
        'translated_answer': translated_answers
    }

dataset = load_dataset("Slim205/wiki_data_more_2_filtered")
subset = dataset['train'].select(range(50))

translated_dataset = subset.map(translate_sample_batch, batched=True, batch_size=2)

translated_dataset.push_to_hub('Slim205/translated_wikipedia_10k_test2')
