import wikipedia as w

def save_list_to_file(filename, data_list):
    with open(filename, 'w') as f:
        for item in data_list:
            f.write("%s\n" % item)

titles_level0 = [
    'Algeria', 'Ancient Egypt', 'Islamic art',
   'Arab culture',
    'Islamic funeral', 'Geography of the Arab world',
    'History of the Arabs', 'Arabic', 'Arabic literature',  'Arabic music', 'Islamic ornament', 'Islamic philosophy',
    'Science in the medieval Islamic world', 'Arab wedding', 'Bahrain', 'Comoros',
    'History of modern Egypt', 'Iraq', 'Islam', 'Jordan', 'Kuwait', 'Lebanon', 'Libya', 'Mauritania', 'History of Mesopotamia',
    'Morocco', 'Oman', 'State of Palestine', 'Qatar', 'Saudi Arabia', 'Somalia', 'Sudan',
    'Syria', 'Tunisia', 'United Arab Emirates', 'Yemen'
]


titles_level1 = []

for i in range(len(titles_level0)) : 
    r= w.WikipediaPage(titles_level0[i])
    titles_level1.extend(r.links)

print(f"End of level 1 with {len(titles_level1)} links")
save_list_to_file('titles_level1.txt', titles_level1)

title_level_1_filtered = []
titles_level2 = []
s = 0
for i in range(len(titles_level1)) : 
    print(i)
    try : 
        r= w.WikipediaPage(titles_level1[i])
        titles_level2.extend(r.links)
        title_level_1_filtered.append(titles_level1[i])
    except Exception as e:
        print(titles_level1[i])
        s+=1

final_list1 = list(set(title_level_1_filtered + titles_level0))
save_list_to_file('titles_level1_filtered.txt', final_list1)

print("The percentage of found titles with errors: ", round(s/len(titles_level1)*100, 2))
print(f"End of level 2 with {len(titles_level2)} links")
save_list_to_file('titles_level2.txt', titles_level2)

title_level_2_filtered = []
s = 0
for i in range(len(titles_level2)) : 
    print(i)
    try : 
        r= w.WikipediaPage(titles_level2[i])
        title_level_2_filtered.append(titles_level2[i])
    except Exception as e:
        print(titles_level2[i])
        s+=1
final_list2 = list(set(title_level_2_filtered + final_list1))
save_list_to_file('titles_level1_filtered.txt', final_list2)
