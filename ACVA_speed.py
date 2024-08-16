import wikipedia as w
from concurrent.futures import ThreadPoolExecutor, as_completed

def save_list_to_file(filename, data_list):
    with open(filename, 'w') as f:
        for item in data_list:
            f.write("%s\n" % item)

def fetch_links(title):
    try:
        page = w.WikipediaPage(title)
        return page.links
    except Exception as e:
        print(f"Error with page: {title}")
        return []

ACVA_SUBSETS = [
    "Algeria", "Ancient_Egypt", "Arab_Empire", "Arabic_Architecture", "Arabic_Art", "Arabic_Astronomy",
    "Arabic_Calligraphy", "Arabic_Ceremony", "Arabic_Clothing", "Arabic_Culture", "Arabic_Food",
    "Arabic_Funeral", "Arabic_Geography", "Arabic_History", "Arabic_Language_Origin", "Arabic_Literature",
    "Arabic_Math", "Arabic_Medicine", "Arabic_Music", "Arabic_Ornament", "Arabic_Philosophy",
    "Arabic_Physics_and_Chemistry", "Arabic_Wedding", "Bahrain", "Comoros", "Egypt_modern",
    "InfluenceFromAncientEgypt", "InfluenceFromByzantium", "InfluenceFromChina", "InfluenceFromGreece",
    "InfluenceFromIslam", "InfluenceFromPersia", "InfluenceFromRome", "Iraq", "Islam_Education",
    "Islam_branches_and_schools", "Islamic_law_system", "Jordan", "Kuwait", "Lebanon", "Libya", "Mauritania",
    "Mesopotamia_civilization", "Morocco", "Oman", "Palestine", "Qatar", "Saudi_Arabia", "Somalia",
    "Sudan", "Syria", "Tunisia", "United_Arab_Emirates", "Yemen", "communication", "computer_and_phone",
    "daily_life", "entertainment"
]

# Collect the titles from Wikipedia
titles_level0 = []
for subject in ACVA_SUBSETS:
    titles_level0.extend(w.search(subject, results=10))

# Save the level 0 titles
save_list_to_file('titles_level0.txt', titles_level0)
print(f"End of level 0 with {len(titles_level0)} links")

# Fetch level 1 links in parallel
titles_level1 = []
with ThreadPoolExecutor(max_workers=10) as executor:
    future_to_title = {executor.submit(fetch_links, title): title for title in titles_level0}
    for future in as_completed(future_to_title):
        links = future.result()
        titles_level1.extend(links)

# Save the level 1 titles
save_list_to_file('titles_level1.txt', titles_level1)
print(f"End of level 1 with {len(titles_level1)} links")

# Fetch level 2 links in parallel
titles_level2 = []
with ThreadPoolExecutor(max_workers=10) as executor:
    future_to_title = {executor.submit(fetch_links, title): title for title in titles_level1}
    for future in as_completed(future_to_title):
        links = future.result()
        titles_level2.extend(links)

# Save the level 2 titles
save_list_to_file('titles_level2.txt', titles_level2)
print(f"End of level 2 with {len(titles_level2)} links")
