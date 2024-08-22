from datasets import load_dataset, DatasetDict, Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import wikipediaapi

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
    return l 


def process_title(title, wiki_wiki):
    page = wiki_wiki.page(title)
    l = get_list_titles(page)
    l = list(set(l))
    list_text = []

    for title in l:
        if title not in ['See also', 'References', 'External links']:
            text_title = page.section_by_title(title).text
            if len(text_title) > 100 : 
                list_text.append((text_title,page.fullurl, page.title))

    return list_text


def main():
    titles = load_list_from_file('wiki_list_level1.txt')
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

    results=[]
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_title, title, wiki_wiki) for title in titles]
        for future in tqdm(as_completed(futures), total=len(futures)):
            results_int = future.result()
            results.extend(results_int)
    print("====================================================================\n")
    print("Number of passages : ",len(results))
    results.sort(key= lambda x: len(x[0]))
    index_80 = int(.8*len(results))-6

    data_to_keep = results[index_80:]

    print("====================================================================\n")
    print("Length of the first element in the top 20% : ",len(results[index_80][0]))
    print("====================================================================\n")
    print("Length of the last element not in the top 20% : ",len( results[index_80-1][0]))
    print("====================================================================\n")

    list_passages = []
    list_links = []
    list_titles= []
    for x in data_to_keep : 
        list_passages.append(x[0])
        list_links.append(x[1])
        list_titles.append(x[2])
    data = {'title':list_titles,'passage' : list_passages,'link':list_links}
    dataset = Dataset.from_dict(data)
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict = dataset_dict.shuffle(seed=42)

    output_path = "./top20_wiki_data"
    dataset_dict.save_to_disk(output_path)
    print(f"Translated dataset saved to {output_path}")

    repo_name = "Slim205/top20_wiki_data"
    dataset_dict.push_to_hub(repo_name)
    print(f"Translated dataset saved and pushed to Hugging Face repo: {repo_name}")

if __name__ == '__main__':
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent='MyProjectName (merlin@example.com)',
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )
    main()
