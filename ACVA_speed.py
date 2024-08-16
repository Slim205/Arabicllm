import wikipedia as w
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def save_list_to_file(filename, data_list):
    with open(filename, 'w') as f:
        for item in data_list:
            f.write("%s\n" % item)

def fetch_links(title):
    try:
        page = w.WikipediaPage(title)
        return title, page.links
    except w.exceptions.DisambiguationError as e:
        try:
            first_option_page = w.WikipediaPage(e.options[0])
            return e.options[0], first_option_page.links
        except Exception as e:
            print(f"Error fetching links for disambiguation option")
            return title, []
    except Exception as e:
        print(f"Error fetching links for {title}: {e}")
        return title, []

# Level 0 titles
titles_level0 = [
   'Arab culture'
]
# Fetch level 1 links using multi-threading with max_workers=10 and tqdm progress bar
titles_level1 = []
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(tqdm(executor.map(fetch_links, titles_level0), total=len(titles_level0), desc="Fetching Level 1 Links"))

for title, links in results:
    titles_level1.extend(links)

titles_level1 = list(set(titles_level1))

print(f"End of level 1 with {len(titles_level1)} links")
save_list_to_file('titles_level1.txt', titles_level1)

# Filter level 1 and fetch level 2 links using multi-threading with max_workers=10 and tqdm progress bar
title_level_1_filtered = []
titles_level2 = []

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(tqdm(executor.map(fetch_links, titles_level1), total=len(titles_level1), desc="Fetching Level 2 Links"))

for title, links in results:
    if links:
        title_level_1_filtered.append(title)
        titles_level2.extend(links)

titles_level2 = list(set(titles_level2))

final_list1 = list(set(title_level_1_filtered + titles_level0))
save_list_to_file('titles_level1_filtered.txt', final_list1)

print(f"End of level 2 with {len(titles_level2)} links")
save_list_to_file('titles_level2.txt', titles_level2)

# Filter level 2 links using multi-threading with max_workers=10 and tqdm progress bar
title_level_2_filtered = []

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(tqdm(executor.map(fetch_links, titles_level2), total=len(titles_level2), desc="Filtering Level 2 Links"))

for title, links in results:
    if links:
        title_level_2_filtered.append(title)

final_list2 = list(set(title_level_2_filtered + final_list1))
save_list_to_file('titles_level2_filtered.txt', final_list2)
