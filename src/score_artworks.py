"""
Task: Score MET aesthetic concept data with hedonometer
This script loads the Eastern/Western aesthetic dataset and applies labMT scoring
"""

import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
from collections import Counter

print("="*60)
print("HEDONOMETER SCORING: MET AESTHETIC CONCEPTS")
print("="*60)

# -----------------------------------------------------------------------------
# Set up paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Load MET aesthetic data
# -----------------------------------------------------------------------------
print("\n Loading MET aesthetic data...")
met_path = RAW_DIR / "met_aesthetic_raw.csv"

if not met_path.exists():
    print(f"Error: File not found at {met_path}")
    print("Please run met_fetch.py first")
    exit()

df = pd.read_csv(met_path)
print(f"Loaded {len(df)} artworks")
print(f"Categories: {df['category'].value_counts().to_dict()}")

# -----------------------------------------------------------------------------
# Load labMT word list
# -----------------------------------------------------------------------------
print("\n Loading labMT word list...")
labMT_path = PROCESSED_DIR / "labMT_cleaned.csv"

if labMT_path.exists():
    labMT_df = pd.read_csv(labMT_path)
    labMT_dict = dict(zip(labMT_df['word'], labMT_df['happiness_average']))
    print(f"Loaded {len(labMT_dict)} words from labMT")
    
    # Show example words
    example_words = ['love', 'beauty', 'death', 'peace', 'joy', 'sorrow']
    print("\nExample words from labMT:")
    for word in example_words:
        if word in labMT_dict:
            print(f"  {word}: {labMT_dict[word]:.2f}")
else:
    print("Error: labMT_cleaned.csv not found in data/processed/")
    exit()

# -----------------------------------------------------------------------------
# Define tokenization and scoring functions
# -----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Clean text by:
    1. Converting to lowercase
    2. Removing punctuation
    3. Removing extra whitespace
    
    Tokenization strategy documented here:
    - Words are split on whitespace after cleaning
    - This follows standard NLP preprocessing
    - No stop words are removed to maintain natural language distribution
    - All words are kept, including neutral ones (scores around 5)
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation (keep only letters, numbers, and spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def score_text(text: str, word_dict: dict) -> dict:
    """
    Calculate happiness score for a piece of text.
    
    Scoring formula (Dodds et al. 2011):
    happiness = (sum of scores for matched words) / (number of matched words)
    
    Methodological choices:
    - Words not in labMT are IGNORED (do not affect score)
    - Repeated words are counted multiple times (follows formula)
    - Coverage is reported as matched_words / total_words
    - No words are excluded based on neutrality
    """
    if not isinstance(text, str) or len(text) == 0:
        return {'score': None, 'matched': 0, 'total': 0, 'coverage': 0.0}
    
    # Clean the text
    cleaned = clean_text(text)
    
    # Tokenization: split on whitespace
    words = cleaned.split()
    total = len(words)
    
    if total == 0:
        return {'score': None, 'matched': 0, 'total': 0, 'coverage': 0.0}
    
    # Match words to labMT dictionary
    scores = []
    matched_words = []
    for word in words:
        if word in word_dict:
            scores.append(word_dict[word])
            matched_words.append(word)
    
    matched = len(scores)
    
    # Handle case with no matches
    if matched == 0:
        return {
            'score': None,
            'matched': 0,
            'total': total,
            'coverage': 0.0,
            'matched_words': []
        }
    
    # Calculate average score and coverage
    return {
        'score': sum(scores) / matched,
        'matched': matched,
        'total': total,
        'coverage': matched / total,
        'matched_words': matched_words
    }

# -----------------------------------------------------------------------------
# Apply scoring to all titles
# -----------------------------------------------------------------------------
print("\nScoring artwork titles with hedonometer...")

results = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
    text = row.get('title', '')
    results.append(score_text(text, labMT_dict))

# Add results to dataframe
results_df = pd.DataFrame(results)
df_scored = pd.concat([df, results_df], axis=1)

# -----------------------------------------------------------------------------
# Summary statistics
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("SCORING SUMMARY")
print("="*60)

scored = df_scored['score'].notna().sum()
print(f"Artworks with scores: {scored}/{len(df_scored)} ({scored/len(df_scored)*100:.1f}%)")

print(f"\nHappiness score statistics:")
print(df_scored['score'].describe())

print(f"\nCoverage statistics:")
print(f"  Mean coverage: {df_scored['coverage'].mean():.3f}")
print(f"  Median coverage: {df_scored['coverage'].median():.3f}")
print(f"  Artworks with no matches: {(df_scored['coverage'] == 0).sum()}")

# -----------------------------------------------------------------------------
# Eastern vs Western comparison (basic, no bootstrap)
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("EASTERN VS WESTERN COMPARISON")
print("="*60)

eastern = df_scored[df_scored['category'] == 'eastern']['score'].dropna()
western = df_scored[df_scored['category'] == 'western']['score'].dropna()

print(f"\nEastern (n={len(eastern)}): mean={eastern.mean():.3f}, std={eastern.std():.3f}")
print(f"Western (n={len(western)}): mean={western.mean():.3f}, std={western.std():.3f}")

# -----------------------------------------------------------------------------
# OOV Analysis: Identify common words not in labMT
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("OOV ANALYSIS")
print("="*60)

# Collect all words from all titles
all_words = []
for text in df['title'].dropna():
    if isinstance(text, str):
        words = clean_text(text).split()
        all_words.extend(words)

# Count word frequencies
word_freq = Counter(all_words)

# Find OOV words (not in labMT)
oov_words = [(word, freq) for word, freq in word_freq.most_common(50) 
             if word not in labMT_dict]

print("\nTop 10 most common words NOT in labMT dictionary:")
print("(These are typically proper nouns, foreign terms, or domain-specific vocabulary)")
for word, freq in oov_words[:10]:
    print(f"  '{word}': appears {freq} times")

# -----------------------------------------------------------------------------
# Save scored data
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("SAVING SCORED DATA")
print("="*60)

output_path = PROCESSED_DIR / "met_aesthetic_scored.csv"
df_scored.to_csv(output_path, index=False)
print(f"Scored data saved to: {output_path}")

# Save summary statistics
summary = pd.DataFrame({
    'metric': ['total_artworks', 'scored_artworks', 'mean_happiness', 
               'median_happiness', 'mean_coverage', 'eastern_mean', 'western_mean'],
    'value': [
        len(df_scored),
        scored,
        df_scored['score'].mean(),
        df_scored['score'].median(),
        df_scored['coverage'].mean(),
        eastern.mean(),
        western.mean()
    ]
})
summary.to_csv(TABLES_DIR / "met_aesthetic_summary.csv", index=False)
print(f"Summary saved to: {TABLES_DIR / 'met_aesthetic_summary.csv'}")

# Save OOV analysis
oov_df = pd.DataFrame(oov_words[:30], columns=['word', 'frequency'])
oov_df.to_csv(TABLES_DIR / "met_aesthetic_oov.csv", index=False)
print(f"OOV analysis saved to: {TABLES_DIR / 'met_aesthetic_oov.csv'}")

print("\n" + "="*60)
print("SCORING COMPLETE!")
print("="*60)
print("\nFiles created:")
print(f"  - {PROCESSED_DIR / 'met_aesthetic_scored.csv'}")
print(f"  - {TABLES_DIR / 'met_aesthetic_summary.csv'}")
print(f"  - {TABLES_DIR / 'met_aesthetic_oov.csv'}")