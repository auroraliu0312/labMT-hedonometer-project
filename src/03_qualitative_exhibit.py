"""
Script 3: Qualitative Exploration - Building a Word Exhibit
Run this after data_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*50)
print("QUALITATIVE EXPLORATION: WORD EXHIBIT")
print("="*50)

# Find the data file - try different possible locations
possible_paths = [
    'data/labMT_cleaned.csv',
    '../data/labMT_cleaned.csv',
    'data/raw/Data_Set_S1.txt',
    '../data/raw/Data_Set_S1.txt'
]

df = None
for path in possible_paths:
    try:
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(path, sep='\t', skiprows=3, na_values='--')
        print(f"✅ Loaded data from: {path}")
        print(f"Shape: {df.shape}")
        break
    except FileNotFoundError:
        continue

if df is None:
    print("❌ Could not find data file. Please check the path.")
    exit()

# Create a function to check corpus presence
def get_corpus_presence(row):
    corpora = []
    if pd.notna(row.get('twitter_rank')):
        corpora.append('Twitter')
    if pd.notna(row.get('google_rank')):
        corpora.append('Google')
    if pd.notna(row.get('nyt_rank')):
        corpora.append('NYT')
    if pd.notna(row.get('lyrics_rank')):
        corpora.append('Lyrics')
    return ', '.join(corpora) if corpora else 'None'

# Add corpus presence column
df['corpus_presence'] = df.apply(get_corpus_presence, axis=1)

# 1. Select 5 VERY POSITIVE words (highest happiness scores)
positive_words = df.nlargest(5, 'happiness_average')[
    ['word', 'happiness_average', 'happiness_standard_deviation', 'corpus_presence']
].copy()
positive_words['category'] = 'Very Positive'

# 2. Select 5 VERY NEGATIVE words (lowest happiness scores)
negative_words = df.nsmallest(5, 'happiness_average')[
    ['word', 'happiness_average', 'happiness_standard_deviation', 'corpus_presence']
].copy()
negative_words['category'] = 'Very Negative'

# 3. Select 5 HIGHLY CONTESTED words (highest standard deviation)
contested_words = df.nlargest(5, 'happiness_standard_deviation')[
    ['word', 'happiness_average', 'happiness_standard_deviation', 'corpus_presence']
].copy()
contested_words['category'] = 'Highly Contested'

# 4. Select 5 INTERESTING/CULTURALLY LOADED words
interesting_candidates = [
    'gay', 'queer', 'feminist', 'trump', 'obama', 'covid', 'pandemic',
    'thou', 'thee', 'hath', 'internet', 'google', 'tweet', 'selfie',
    'hipster', 'millennial', 'boomer', 'woke', 'lit', 'sick', 'cool'
]

available_interesting = df[df['word'].isin(interesting_candidates)].copy()
if len(available_interesting) < 5:
    additional = df[
        (df['happiness_standard_deviation'] > 1.5) & 
        (df['twitter_rank'].notna()) & 
        (df['google_rank'].isna())
    ].head(5)
    interesting_words = pd.concat([available_interesting, additional]).head(5)
else:
    interesting_words = available_interesting.head(5)

interesting_words['category'] = 'Interesting/Culturally Loaded'
interesting_words = interesting_words[
    ['word', 'happiness_average', 'happiness_standard_deviation', 'corpus_presence', 'category']
]

# Combine all exhibits
exhibit = pd.concat([
    positive_words, 
    negative_words, 
    contested_words, 
    interesting_words
])

exhibit = exhibit.reset_index(drop=True)

print("\n" + "="*50)
print("WORD EXHIBIT: 20 Words for Close Reading")
print("="*50)
print("\n")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 20)

print(exhibit.to_string(index=False))

# Save to CSV
exhibit.to_csv('data/word_exhibit.csv', index=False)
print("\n✅ Exhibit saved to data/word_exhibit.csv")




