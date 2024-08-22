from datasets import load_dataset
import re

# Load the dataset
ref = "Slim205/wiki_translated_gemma"
dataset = load_dataset(ref)

def add_id_column(example, idx):
    """
    Add an 'id' column to each example in the dataset.
    
    Args:
        example (dict): A dictionary representing a row in the dataset.
        idx (int): The index of the row in the dataset.
    
    Returns:
        dict: The updated row with the added 'id' field.
    """
    example['id'] = idx
    return example

# Add an 'id' column to each split
#dataset_with_id = dataset.map(add_id_column, with_indices=True)

def contains_english(text):
    """
    Check if a given text contains any English letters.
    
    Args:
        text (str): The text to check.
    
    Returns:
        bool: True if the text contains English letters, False otherwise.
    """
    english_pattern = re.compile(r'[a-zA-Z]')
    return bool(english_pattern.search(text))

def contains_japanese(text):
    """
    Check if a given text contains any Japanese characters (Hiragana, Katakana, or Kanji).
    
    Args:
        text (str): The text to check.
    
    Returns:
        bool: True if the text contains Japanese characters, False otherwise.
    """
    japanese_pattern = re.compile(r'[\u3040-\u30FF\u4E00-\u9FAF]')
    return bool(japanese_pattern.search(text))

def contains_hebrew(text):
    """
    Check if a given text contains any Hebrew characters.
    
    Args:
        text (str): The text to check.
    
    Returns:
        bool: True if the text contains Hebrew characters, False otherwise.
    """
    hebrew_pattern = re.compile(r'[\u0590-\u05FF]')
    return bool(hebrew_pattern.search(text))

def ends_with_question_mark(sentence):
    """
    Check if the given sentence ends with a question mark.
    
    Args:
        sentence (str): The sentence to check.
    
    Returns:
        bool: True if the sentence ends with a question mark, False otherwise.
    """
    return sentence.strip().endswith('؟')

def is_length_less_than_663(text):
    """
    Check if the length of the given text is less than 663 characters.
    
    Args:
        text (str): The text to check.
    
    Returns:
        bool: True if the length of the text is less than 663 characters, False otherwise.
    """
    return len(text) < 663

def is_length_less_than_592(text):
    """
    Check if the length of the given text is less than 592 characters.
    
    Args:
        text (str): The text to check.
    
    Returns:
        bool: True if the length of the text is less than 592 characters, False otherwise.
    """
    return len(text) < 592

def filter_all_conditions(sample):
    """
    Apply all filtering conditions to a sample.
    
    Args:
        sample (dict): A dictionary representing a row in the dataset.
    
    Returns:
        bool: True if the sample passes all conditions, False otherwise.
    """
    return (
        not (contains_english(sample['answer']) or contains_english(sample['question'])) and
        not (contains_japanese(sample['answer']) or contains_japanese(sample['question'])) and
        not (contains_hebrew(sample['answer']) or contains_hebrew(sample['question'])) 
       # ends_with_question_mark(sample['question']) and
       # is_length_less_than_663(sample['question']) and
       # is_length_less_than_592(sample['answer'])
    )

# Apply all filters in a single pass
dataset_filtered = dataset.filter(filter_all_conditions)

# Push the final filtered dataset to the hub
dataset_filtered.push_to_hub("Slim205/filtered-wiki-translated-gemma")
