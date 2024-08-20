from vllm import LLM,SamplingParams
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import wikipediaapi 
import fire

def load_list_from_file(filename):
    with open(filename, 'r') as f:
        data_list = [line.strip() for line in f]
    return data_list

def store_title(section,current_list_titles) : 
    if len(section.sections) == 0 : 
        current_list_titles.append(section.title)
    else : 
        for s in section.sections : 
            store_title(s,current_list_titles)
    return current_list_titles

def get_list_titles(page) :
    l = []
    for section in page.sections : 
        l = store_title(section,l)
   # pos = l.index('See also')
    #l = l[:pos]
    return l 

def get_data(text,tokenizer) : 
    prompt = f"""
Read the text below and create a question-answer pair based on it. Separate the question and answer with "$".

{text}

Provide only the question and answer, without any tags, comments, or references to the input text.
"""
    
    messages = [{"role": "user", "content":prompt}]
    prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
    return prompt_with_chat_template

def wiki( model_name : str,repo_name : str):
    titles = load_list_from_file('wikipedia_links_list.txt')
    titles = titles[:50]
    titles = [
    'Algeria', 'Ancient Egypt', 'Caliphate', 'Islamic architecture', 'Islamic art',
    'Astronomy in the medieval Islamic world', 'Arabic calligraphy', 'Arab culture',
    'Arabs', 'Arab cuisine', 'Islamic funeral', 'Geography of the Arab world',
    'History of the Arabs', 'Arabic', 'Arabic literature', 'Mathematics in the medieval Islamic world',
    'Medicine in the medieval Islamic world', 'Arabic music', 'Islamic ornament', 'Islamic philosophy',
    'Science in the medieval Islamic world', 'Arab wedding', 'Bahrain', 'Comoros',
    'History of modern Egypt', 'Iraq', 'Education in Islam', 'Islamic schools and branches',
    'Sharia', 'Jordan', 'Kuwait', 'Lebanon', 'Libya', 'Mauritania', 'History of Mesopotamia',
    'Morocco', 'Oman', 'State of Palestine', 'Qatar', 'Saudi Arabia', 'Somalia', 'Sudan',
    'Syria', 'Tunisia', 'United Arab Emirates', 'Yemen'
]

    #model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(titles[0])
    llm = LLM(model_name,max_model_len=4096) 
    list_passages = []
    list_titles =[]
    list_links = []
    prompt = []
    for title in titles :
        page = wiki_wiki.page(title)
        if 'ar' in page.langlinks : 
            page = page.langlinks['ar']
            l = get_list_titles(page)
            list_text = []
            for title in l : 
                text_title = page.section_by_title(title).text
                if len(text_title) > 100 : 
                    list_text.append(text_title)
        #  list_text.sort(key=lambda x: len(x), reverse=True)
            for text in list_text: #[:10] :
                prompt.append(get_data(text,tokenizer))
                list_passages.append(text)
                list_links.append(page.fullurl)
                list_titles.append(page.title)
    print(prompt[0])

    sampling_params = SamplingParams(max_tokens=512,temperature=0.8, top_p=0.95)
    outputs = llm.generate(prompt,sampling_params)
    llm_questions = []
    llm_answers=[]
    llm_passage = []
    llm_titles = []
    llm_links = []
    for i, item in enumerate(outputs):
        output = item.outputs[0].text
        if '$' in output : 
            llm_questions.append(output.split('$')[0])
            llm_answers.append(output.split('$')[1])
            llm_passage.append(list_passages[i])
            llm_titles.append(list_titles[i])
            llm_links.append(list_links[i])

    data = { "title": llm_titles,"passage" : llm_passage , "question": llm_questions, "answer" : llm_answers, 'link':llm_links}
    dataset = Dataset.from_dict(data)
    dataset_dict = DatasetDict({"train": dataset})

   # dataset_dict.save_to_disk(output_path)
   # print(f"Translated dataset saved to {output_path}")

    dataset_dict.push_to_hub(repo_name)
    print(f"Translated dataset saved and pushed to Hugging Face repo: {repo_name}")
 


if __name__ == '__main__':
    wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='MyProjectName (merlin@example.com)',
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
)
    fire.Fire(wiki)
