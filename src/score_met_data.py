"""
Task: Apply hedonometer to MET artwork titles
This script loads MET data and scores each artwork's title
"""

import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm

print("="*60)
print("Hedonometer Scoring for MET Artworks")
print("="*60)

# -----------------------------------------------------------------------------
# Set up paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Load MET data
# -----------------------------------------------------------------------------
print("\nLoading MET data...")
met_files = list(PROCESSED_DIR.glob("met_artworks_multi_*.csv"))
if not met_files:
    print("Error: No MET data found")
    exit()

# Get most recent file
met_path = sorted(met_files)[-1]
df = pd.read_csv(met_path)
print(f"Loaded {len(df)} artworks")
print(f"File: {met_path.name}")

# -----------------------------------------------------------------------------
# Load labMT word list
# -----------------------------------------------------------------------------
print("\nLoading labMT word list...")
labMT_path = PROCESSED_DIR / "labMT_cleaned.csv"

if labMT_path.exists():
    labMT_df = pd.read_csv(labMT_path)
    labMT_dict = dict(zip(labMT_df['word'], labMT_df['happiness_average']))
    print(f"Loaded {len(labMT_dict)} words from labMT")
    
    # Show some example words
    example_words = ['love', 'death', 'war', 'peace', 'beauty', 'sorrow']
    print("\nExample words from labMT:")
    for word in example_words:
        if word in labMT_dict:
            print(f"  {word}: {labMT_dict[word]:.2f}")
else:
    print("Warning: labMT file not found. Using fallback dictionary")
    labMT_dict = {
        'love': 8.42, 'happy': 8.30, 'great': 7.50, 'good': 6.80,
        'bad': 2.50, 'hate': 2.30, 'death': 1.54, 'war': 2.50,
        'peace': 7.20, 'beauty': 7.80, 'sorrow': 2.30, 'joy': 8.16,
        'flower': 6.50, 'nature': 6.50, 'art': 6.00, 'painting': 5.20
    }

# -----------------------------------------------------------------------------
# Define tokenization and scoring functions
# -----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Clean text by:
    1. Converting to lowercase
    2. Removing punctuation
    3. Removing extra whitespace
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text

def score_text(text: str, word_dict: dict) -> dict:
    """
    Calculate happiness score for a piece of text.
    
    Tokenization strategy:
    - Split on whitespace after cleaning
    - Each word is a token
    - Match tokens to labMT dictionary
    """
    if not isinstance(text, str) or len(text) == 0:
        return {'score': None, 'matched': 0, 'total': 0, 'coverage': 0.0}
    
    cleaned = clean_text(text)
    words = cleaned.split()
    total = len(words)
    
    if total == 0:
        return {'score': None, 'matched': 0, 'total': 0, 'coverage': 0.0}
    
    scores = []
    matched_words = []
    for word in words:
        if word in word_dict:
            scores.append(word_dict[word])
            matched_words.append(word)
    
    matched = len(scores)
    
    if matched == 0:
        return {
            'score': None,
            'matched': 0,
            'total': total,
            'coverage': 0.0,
            'matched_words': []
        }
    
    return {
        'score': sum(scores) / matched,
        'matched': matched,
        'total': total,
        'coverage': matched / total,
        'matched_words': matched_words
    }

# -----------------------------------------------------------------------------
# Apply scoring to analysis_text
# -----------------------------------------------------------------------------
print("\nScoring artwork titles...")

results = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
    text = row.get('analysis_text', '')
    results.append(score_text(text, labMT_dict))

# Add results to dataframe
results_df = pd.DataFrame(results)
df_scored = pd.concat([df, results_df], axis=1)

# -----------------------------------------------------------------------------
# Summary statistics
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("Scoring Summary")
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
# Show examples
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("Example Scored Artworks")
print("="*60)

# High scoring titles
print("\n High happiness score examples (score > 7.0):")
high = df_scored[df_scored['score'] > 7.0].head(3)
if len(high) > 0:
    for idx, row in high.iterrows():
        print(f"\n  Score: {row['score']:.2f}")
        print(f"  Title: {row['title']}")
        print(f"  Artist: {row.get('artistDisplayName', 'Unknown')}")
        print(f"  Matched words: {row.get('matched_words', [])}")
else:
    print("  No high-scoring artworks found")

# Low scoring titles
print("\n Low happiness score examples (score < 4.0):")
low = df_scored[df_scored['score'] < 4.0].head(3)
if len(low) > 0:
    for idx, row in low.iterrows():
        print(f"\n  Score: {row['score']:.2f}")
        print(f"  Title: {row['title']}")
        print(f"  Artist: {row.get('artistDisplayName', 'Unknown')}")
        print(f"  Matched words: {row.get('matched_words', [])}")
else:
    print("  No low-scoring artworks found")

# -----------------------------------------------------------------------------
# Analysis by search term
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("Analysis by Search Term")
print("="*60)

if 'search_term' in df_scored.columns:
    term_stats = df_scored.groupby('search_term')['score'].agg(['count', 'mean', 'std', 'median']).round(3)
    term_stats = term_stats.sort_values('mean', ascending=False)
    print("\n" + term_stats.to_string())
    term_stats.to_csv(TABLES_DIR / "met_scores_by_search_term.csv")

# -----------------------------------------------------------------------------
# Analysis by department
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("Analysis by Department")
print("="*60)

if 'department' in df_scored.columns:
    dept_stats = df_scored.groupby('department')['score'].agg(['count', 'mean', 'std']).round(3)
    dept_stats = dept_stats.sort_values('mean', ascending=False)
    dept_stats = dept_stats[dept_stats['count'] >= 3]  # Only departments with at least 3 artworks
    print("\n" + dept_stats.to_string())
    dept_stats.to_csv(TABLES_DIR / "met_scores_by_department.csv")

# -----------------------------------------------------------------------------
# Save scored data
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("Saving Scored Data")
print("="*60)

output_path = PROCESSED_DIR / "met_artworks_scored.csv"
df_scored.to_csv(output_path, index=False)
print(f"Scored data saved to: {output_path}")

# Save summary
summary = pd.DataFrame({
    'metric': ['total_artworks', 'scored_artworks', 'mean_happiness', 'median_happiness', 'mean_coverage'],
    'value': [
        len(df_scored),
        scored,
        df_scored['score'].mean(),
        df_scored['score'].median(),
        df_scored['coverage'].mean()
    ]
})
summary.to_csv(TABLES_DIR / "met_scoring_summary.csv", index=False)

print("\n" + "="*60)
print("MET Scoring Complete!")
print("="*60)

# -----------------------------------------------------------------------------
# OOV Analysis: Identify common words not in labMT
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("OOV Analysis")
print("="*60)

from collections import Counter

# Collect all words from all artworks
all_words = []
for text in df['analysis_text'].dropna():
    if isinstance(text, str):
        words = clean_text(text).split()
        all_words.extend(words)

# Count word frequencies
word_freq = Counter(all_words)

# Find OOV words (not in labMT)
oov_words = [(word, freq) for word, freq in word_freq.most_common(50) 
             if word not in labMT_dict]

print("\nTop 10 most common words NOT in labMT dictionary:")
for word, freq in oov_words[:10]:
    print(f"  '{word}': appears {freq} times")

print("\nThese OOV words are typically art-specific terms (titles, mediums, etc.)")
print("that are not covered by the general-purpose labMT lexicon.")

# Save OOV analysis
oov_df = pd.DataFrame(oov_words[:30], columns=['word', 'frequency'])
oov_df.to_csv(TABLES_DIR / "met_oov_words.csv", index=False)
print(f"\nOOV analysis saved to: {TABLES_DIR / 'met_oov_words.csv'}")