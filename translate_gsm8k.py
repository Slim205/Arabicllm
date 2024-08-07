import fire
from vllm import LLM,SamplingParams
from datasets import load_dataset, DatasetDict,Dataset

def translate_gsm8k(model_name: str, target_language: str = 'Arabic', output_path: str = './translated_gsm8k', repo_name: str = 'Slim205/translated-gsm8k'):

    dataset = load_dataset("openai/gsm8k", "main")
    dataset = dataset['train'].select(range(20))
    prompts = []

    def apply_chat_template_question(text) : 
        example0= "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?"
        response0="جيمس يكتب رسالة مكونة من 3 صفحات إلى صديقين مختلفين مرتين في الأسبوع. كم عدد الصفحات التي يكتبها في السنة؟"
        example1= "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"
        response1="وينغ تكسب 12 دولاراً في الساعة مقابل رعاية الأطفال. بالأمس، قامت برعاية الأطفال لمدة 50 دقيقة فقط. كم كسبت؟"
        example 2 ="There are 290 liters of oil in 24 cans. If 10 of the cans are holding 8 liters each, how much oil is each of the remaining cans holding?"
        response 2 = "هناك 290 لتراً من الزيت في 24 علبة. إذا كانت 10 من العلب تحتوي على 8 لترات لكل منها، فكم كمية الزيت الموجودة في كل واحدة من العلب المتبقية؟"
        example3 = "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?"
        response3 = "جوي تستطيع قراءة 8 صفحات من كتاب في 20 دقيقة. كم ساعة ستستغرق لقراءة 120 صفحة؟"
        example4 = "There are 25 roses in a garden. There are 40 tulips. There are 35 daisies. What percentage of flowers are not roses?"
        response4 = "هناك 25 وردة في الحديقة. هناك 40 زهرة توليب. هناك 35 زهرة أقحوان. ما هي نسبة الزهور التي ليست وروداً؟"
        messages = [
            {"role": "user", "content": f"Translate the following English text to {target_language} : {example0}"},
            {"role": "assistant", "content": response0},
            {"role": "user", "content": f"Translate the following English text to {target_language} : {example1}"},
            {"role": "assistant", "content": response1},
            {"role": "user", "content": f"Translate the following English text to {target_language} : {example2}"},
            {"role": "assistant", "content": response2},
            {"role": "user", "content": f"Translate the following English text to {target_language} : {example3}"},
            {"role": "assistant", "content": response3},
            {"role": "user", "content": f"Translate the following English text to {target_language} : {example4}"},
            {"role": "assistant", "content": response4},
            {"role": "user", "content": f"Translate the following English text to {target_language} : {text}"},
        ]
        prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
        return prompt_with_chat_template

    def apply_chat_template_question_answer(text) : 
        example0 = "Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72"
        response0 = "ناتاليا باعت 48/2 = 24 مشبكاً في مايو. ناتاليا باعت 48 + 24 = 72 مشبكاً إجمالاً في أبريل ومايو. #### 72"
        example1 = "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10"
        response1= "وينغ تكسب 12/60 = 0.2 دولار في الدقيقة. عملت لمدة 50 دقيقة، وكسبت 0.2 × 50 = 10 دولارات. #### 10"
        example2 = "Carlos memorized 24/6=<<24/6=4>>4 digits of pi. Sam memorized 4+6=10 digits of pi. #### 10"
        response2 = "كارلوس حفظ 24/6 = 4 أرقام من باي. سام حفظ 4 + 6 = 10 أرقام من باي. #### 10"
        example3 = "So far, Mary has paid back $15 +$23=$<<15+23=38>>38 of the credit. So she still needs to pay $100-$38=$<<100-38=62>>62 #### 62"
        response3 = "حتى الآن، دفعت ماري 15 + 23 = 38 دولارًا من الائتمان. لذا لا يزال عليها دفع 100 - 38 = 62 دولارًا. #### 62"
        example4 = "Ben collected 36/3=<<36/3=12>>12 shells Alan collected 12*4=<<12*4=48>>48 shells #### 48"
        response4 = "بن جمع 36/3 = 12 قشرة. آلان جمع 12 * 4 = 48 قشرة. #### 48"

        messages = [
            {"role": "user", "content": f"Translate the following English text to {target_language} : {example0}"},
            {"role": "assistant", "content": response0},
            {"role": "user", "content": f"Translate the following English text to {target_language} : {example1}"},
            {"role": "assistant", "content": response1},
            {"role": "user", "content": f"Translate the following English text to {target_language} : {example2}"},
            {"role": "assistant", "content": response2},
            {"role": "user", "content": f"Translate the following English text to {target_language} : {example3}"},
            {"role": "assistant", "content": response3},
            {"role": "user", "content": f"Translate the following English text to {target_language} : {example4}"},
            {"role": "assistant", "content": response4},
            {"role": "user", "content": f"Translate the following English text to {target_language} : {text}"},
        ]
        prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
        return prompt_with_chat_template


    for example in dataset:        
        prompts.append(apply_chat_template_question(example['question']))
        prompts.append(apply_chat_template_answer(example['answer']))
    print(prompts[0])
    if model_name == "Qwen/Qwen2-72B-Instruct" : 
        llm = LLM(model_name, tensor_parallel_size=4,max_model_len=816,gpu_memory_utilization=0.95) 

    elif model_name == "google/gemma-2-27b-it" : 
        llm = LLM(model_name, tensor_parallel_size=2,gpu_memory_utilization=0.8) 

    elif model_name == "meta-llama/Meta-Llama-3.1-70B-Instruct" : 
        llm = LLM(model_name, tensor_parallel_size=4,max_model_len=816,gpu_memory_utilization=0.95) 

    else :
        llm = LLM(model_name) 

    #sampling_params = SamplingParams(max_tokens=100,use_beam_search=True,early_stopping=True,best_of=3,temperature=0)
    sampling_params = SamplingParams(max_tokens=512,temperature=0.8, top_p=0.95)
    outputs = llm.generate(prompts,sampling_params)

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

   # dataset_dict.save_to_disk(output_path)
   # print(f"Translated dataset saved to {output_path}")

    dataset_dict.push_to_hub(repo_name)
    print(f"Translated dataset saved and pushed to Hugging Face repo: {repo_name}")

if __name__ == '__main__':
    fire.Fire(translate_gsm8k)
