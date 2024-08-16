import wikipedia as w

def save_list_to_file(filename, data_list):
    with open(filename, 'w') as f:
        for item in data_list:
            f.write("%s\n" % item)

ACVA_SUBSETS = [
"Algeria", "Ancient_Egypt", "Arab_Empire", "Arabic_Architecture", "Arabic_Art", "Arabic_Astronomy", "Arabic_Calligraphy", "Arabic_Ceremony",
"Arabic_Clothing", "Arabic_Culture", "Arabic_Food", "Arabic_Funeral", "Arabic_Geography", "Arabic_History", "Arabic_Language_Origin",
"Arabic_Literature", "Arabic_Math", "Arabic_Medicine", "Arabic_Music", "Arabic_Ornament", "Arabic_Philosophy", "Arabic_Physics_and_Chemistry",
"Arabic_Wedding", "Bahrain", "Comoros", "Egypt_modern", "InfluenceFromAncientEgypt", "InfluenceFromByzantium", "InfluenceFromChina",
"InfluenceFromGreece", "InfluenceFromIslam", "InfluenceFromPersia", "InfluenceFromRome", "Iraq", "Islam_Education", "Islam_branches_and_schools",
"Islamic_law_system", "Jordan", "Kuwait", "Lebanon", "Libya", "Mauritania", "Mesopotamia_civilization", "Morocco", "Oman", "Palestine", "Qatar",
"Saudi_Arabia", "Somalia", "Sudan", "Syria", "Tunisia", "United_Arab_Emirates", "Yemen",
"communication", "computer_and_phone", "daily_life", "entertainment"] 

titles_level0=[]
for subject in ACVA_SUBSETS : 
  titles_level0.extend(w.search(subject,results=10))

# Save the lists to text files
save_list_to_file('titles_level0.txt', titles_level0)
print(f"End of level 0 with {len(titles_level0)} links")

titles_level1 = []
s = 0
for i in range(len(titles_level0)) : 
    print(i)
    try : 
        r= w.WikipediaPage(titles_level0[i])
        titles_level1.extend(r.links)
    except Exception as e:
        print(titles_level0[i])
        s+=1
print("The percentage of found titles with errors: ", round(s/len(titles_level0)*100, 2))
print(f"End of level 1 with {len(titles_level1)} links")
save_list_to_file('titles_level1.txt', titles_level1)

titles_level2 = []
s = 0
for i in range(len(titles_level1)) : 
    print(i)
    try : 
        r= w.WikipediaPage(titles_level1[i])
        titles_level2.extend(r.links)
    except Exception as e:
        print(titles_level1[i])
        s+=1
print("The percentage of found titles with errors: ", round(s/len(titles_level1)*100, 2))
print(f"End of level 2 with {len(titles_level2)} links")
save_list_to_file('titles_level2.txt', titles_level2)
