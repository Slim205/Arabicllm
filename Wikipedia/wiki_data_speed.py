from vllm import LLM, SamplingParams
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import wikipediaapi
import fire

# Functions load_list_from_file, store_title, get_list_titles, get_question, get_answer remain unchanged
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

def get_question(text,tokenizer) : 
    prompt = f"""
Read the text below and create a question whose answer is the entire text. Don't start your question with "What".

{text}

Provide only the question, without any tags, comments, or references to the input text.
"""

    messages = [{"role": "user", "content":prompt}]
    prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
    return prompt_with_chat_template

#Provide a detailed answer to the following question based on the provided text. 

def get_answer(text,question,tokenizer) : 
    prompt = f"""
Answer to the following question based on the provided text. Don't answer in a single paragraph. 

Text: {text}
Question : {question}

Respond directly, avoiding any tags, comments, or references to the input text.
"""
    messages = [{"role": "user", "content":prompt}]
    prompt_with_chat_template= tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
    return prompt_with_chat_template


def process_title(title, wiki_wiki, tokenizer, llm):
    page = wiki_wiki.page(title)
    l = get_list_titles(page)
    l = list(set(l))
    list_text = []

    for title in l:
        if title not in ['See also', 'References', 'External links']:
            text_title = page.section_by_title(title).text
            if len(text_title) > 100 and len(text_title) < 2204 : 
                list_text.append(text_title)

    results = []
    for text in list_text:
        question_prompt = get_question(text, tokenizer)
        results.append((text, page.fullurl, page.title, question_prompt))

    return results

def generate_outputs(prompts, llm, sampling_params):
    return llm.generate(prompts, sampling_params)

def wiki(model_name: str, repo_name: str):
    titles = load_list_from_file('wiki_list_level1.txt')
   # titles = titles[:1200]
    titles0 = [
    'Algeria', 'Ancient Egypt', 'Caliphate', 'Islamic architecture', 'Islamic art',
    'Astronomy in the medieval Islamic world', 'Arabic calligraphy', 'Arab culture',
    'Arabs', 'Arab cuisine', 'Islamic funeral', 'Geography of the Arab world',
    'History of the Arabs', 'Arabic', 'Arabic literature', 'Mathematics in the medieval Islamic world',
    'Medicine in the medieval Islamic world', 'Arabic music', 'Islamic ornament', 'Islamic philosophy',
    'Science in the medieval Islamic world', 'Arab wedding', 'Bahrain', 'Comoros',
    'History of modern Egypt', 'Iraq', 'Education in Islam', 'Islamic schools and branches',
    'Sharia', 'Jordan', 'Kuwait', 'Lebanon', 'Libya', 'Mauritania', 'History of Mesopotamia',
    'Morocco', 'Oman', 'State of Palestine', 'Qatar', 'Saudi Arabia', 'Somalia', 'Sudan',
    'Syria', 'Tunisia', 'United Arab Emirates', 'Yemen']
    titles = titles0 + titles
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_name == "google/gemma-2-27b-it":
        llm = LLM(model_name, max_model_len=4096, tensor_parallel_size=4, gpu_memory_utilization=0.8)
    elif model_name == "meta-llama/Meta-Llama-3.1-70B-Instruct":
        llm = LLM(model_name, max_model_len=816, tensor_parallel_size=4, gpu_memory_utilization=0.95)
    else:
        llm = LLM(model_name, max_model_len=4096)

    list_passages = []
    list_titles = []
    list_links = []
    prompts = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_title, title, wiki_wiki, tokenizer, llm) for title in titles]
        for future in tqdm(as_completed(futures), total=len(futures)):
            results = future.result()
            for text, link, title, question_prompt in results:
                list_passages.append(text)
                list_links.append(link)
                list_titles.append(title)
                prompts.append(question_prompt)

    sampling_params = SamplingParams(max_tokens=512, temperature=0.8, top_p=0.95)
    outputs = generate_outputs(prompts, llm, sampling_params)

    llm_questions = []
    llm_answers = []
    llm_passage = []
    llm_titles = []
    llm_links = []

    for i, item in enumerate(outputs):
        output = item.outputs[0].text
        llm_questions.append(output)
        llm_passage.append(list_passages[i])
        llm_titles.append(list_titles[i])
        llm_links.append(list_links[i])

    answer_prompts = [get_answer(llm_passage[i], llm_questions[i], tokenizer) for i in range(len(llm_questions))]
    outputs = generate_outputs(answer_prompts, llm, sampling_params)

    for i, item in enumerate(outputs):
        output = item.outputs[0].text
        llm_answers.append(output)

    data = {"title": llm_titles, "passage": llm_passage, "question": llm_questions, "answer": llm_answers, 'link': llm_links}
    dataset = Dataset.from_dict(data)
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict = dataset_dict.shuffle(seed=42)

    dataset_dict.push_to_hub(repo_name)
    print(f"Translated dataset saved and pushed to Hugging Face repo: {repo_name}")

if __name__ == '__main__':
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent='MyProjectName (merlin@example.com)',
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )
    fire.Fire(wiki)
