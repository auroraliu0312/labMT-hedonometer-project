from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
# These are the only functions in this file. Everything else runs sequentially.
# We use helpers only for repeated patterns (printing sections, saving outputs).

#title must be str, return none for this def
def print_section(title: str) -> None:
    """Print a clear section divider in the terminal."""
    bar = "=" * 90
    print("\n" + bar)
    print(title)
    print(bar)


def save_csv(df: pd.DataFrame, filename: str, index: bool = False) -> None:
    """Save a DataFrame to tables/ and print where it went."""
    out_path = TABLES_DIR / filename
    df.to_csv(out_path, index=index)
    print(f"Saved table: {out_path}")


def save_figure(filename: str, dpi: int = 200) -> None:
    """Save the current matplotlib figure to figures/ and print where it went."""
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=dpi)
    print(f"Saved figure: {out_path}")

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
# We build file paths relative to THIS script, so the code works on any machine
# as long as the folder structure is the same.

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Data_Set_S1.txt"

FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"

FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 1. LOAD, CLEAN, AND DESCRIBE THE DATASET
# -----------------------------------------------------------------------------

print_section("1.1 Load the dataset (Data_Set_S1.txt)")
# pd will create dataframe for csv directly
df = pd.read_csv(
    "data/raw/Data_Set_S1.txt",
    sep="\t",        # <--- file is tab-separated
    skiprows=3,       # <--- skip first 3 lines (metadata)
    na_values="--"    # <--- treat '--' as missing, will be labelled as NaN
)

# Show the first 5 rows to check if loaded correctly
print(df.head())  # <--- comment: preview the first few rows

# Convert numeric columns to numeric types explicitly
numeric_cols = ["happiness_rank", "happiness_average", 
                "happiness_standard_deviation", "twitter_rank", 
                "google_rank", "nyt_rank", "lyrics_rank"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")  #all the errors such as letters will be marked as NaN
    # comment: coerce invalid parsing to NaN

# Check the data types after conversion
print(df.dtypes)  # <--- comment: ensure numeric columns are numeric

# Confirm the number of rows and columns
print("Dataset shape:", df.shape)

print_section("1.2 Data dictionary + missing values")

# List each column and its data type
print("Column names and data types:")
print(df.dtypes)  # <--- comment: display all column names with types

# Count missing values per column
missing_counts = df.isna().sum()
print("\nMissing values per column:")
print(missing_counts)  # <--- comment: count how many NaNs in each column

# Optional: create a structured data dictionary as a DataFrame for easier display, you need to create new dataframe by yourself
data_dict = pd.DataFrame({
    "Column": df.columns,
    "Type": [str(df[col].dtype) for col in df.columns], #meaning show dtypes for all col respectively
    "Missing Values": [df[col].isna().sum() for col in df.columns],
    "Description": [
        "Word being assessed",                  # word
        "Rank based on happiness (1 = happiest)",  # happiness_rank
        "Average happiness score (1-9)",       # happiness_average
        "Standard deviation of happiness",     # happiness_standard_deviation
        "Twitter rank of the word",             # twitter_rank
        "Google Books rank of the word",        # google_rank
        "New York Times rank of the word",      # nyt_rank
        "Lyrics rank of the word"               # lyrics_rank
    ]
})

print("\nData Dictionary:")
print(data_dict)  # <--- comment: show the data dictionary in a table format

print_section("1.3 Sanity checks")

# Check for duplicated words: inside df for boolean results , outside for only the real True wprd
duplicated_words = df[df.duplicated(subset="word")]  # English comment: Check if any words are repeated
print("Duplicated words:\n", duplicated_words)

# Inspect a random sample of 15 rows
sample_rows = df.sample(15, random_state=42)  # English comment: Randomly sample 15 rows to review data
print("Random sample of 15 rows:\n", sample_rows)

# Identify the 10 most positive words
top_10_positive = df.nlargest(10, "happiness_average")[["word", "happiness_average"]]  # English comment: Top 10 highest happiness scores
print("Top 10 positive words:\n", top_10_positive)

# Identify the 10 most negative words
bottom_10_negative = df.nsmallest(10, "happiness_average")[["word", "happiness_average"]]  # English comment: Top 10 lowest happiness scores
print("Top 10 negative words:\n", bottom_10_negative)

# -----------------------------------------------------------------------------
# 2. QUANTITATIVE EXPLORATION
# -----------------------------------------------------------------------------

print_section("2.1 Distribution of happiness_average")

h = df["happiness_average"].dropna()
summary_stats = pd.DataFrame(
    {
        "metric": [
            "count",
            "mean",
            "median",
            "std",
            "p05 (5th percentile)",
            "p95 (95th percentile)",
        ],
        "value": [
            float(h.shape[0]), #row
            float(h.mean()),
            float(h.median()),
            float(h.std()),
            float(h.quantile(0.05)),
            float(h.quantile(0.95)),
        ],
    }
)

print(summary_stats.to_string(index=False)) #save table
save_csv(summary_stats, "happiness_average_summary_stats.csv", index=False)

# Histogram：how many words in each percentile, not cumulative
plt.figure()
plt.hist(h, bins=40)
plt.title("Distribution of happiness_average (labMT 1.0)")
plt.xlabel("happiness_average (1–9)")
plt.ylabel("number of words")
plt.tight_layout()
save_figure("happiness_average_hist.png")
plt.close()

print_section("2.2 Disagreement: happiness_standard_deviation")

show_cols = ["word", "happiness_average", "happiness_standard_deviation"]

# Scatter: happiness score vs standard deviation
plt.figure()
plt.scatter(
    df["happiness_average"], #data for x
    df["happiness_standard_deviation"], #data for y
    s=10,
    alpha=0.35,
)
plt.title("Disagreement vs score: happiness_average vs happiness_standard_deviation")
plt.xlabel("happiness_average")
plt.ylabel("happiness_standard_deviation")
plt.tight_layout()
save_figure("happiness_vs_std_scatter.png")
plt.close()

# Which words do people disagree about most?
most_contested_15 = df.sort_values("happiness_standard_deviation", ascending=False).head(15)[show_cols]
print("Top 15 most 'contested' words (highest standard deviation):")
print(most_contested_15.to_string(index=False))
save_csv(most_contested_15, "top_15_contested_words.csv", index=False)

print_section("2.3 Corpus comparison: rank coverage + overlaps")

corpora = {
        "Twitter": "twitter_rank",
        "GoogleBooks": "google_rank",
        "NYT": "nyt_rank",
        "Lyrics": "lyrics_rank",
    }

# ---------------------------------------
# 1) Count how many LabMT words appear in top 5000
# (rank not missing = appears in top 5000)
# ---------------------------------------

corpus_counts = {}

for name, col in corpora.items():
        count_present = df[col].notna().sum()
        corpus_counts[name] = count_present
        print(f"{name}: {count_present} words present (rank not missing)")

# Bar chart of presence
plt.figure()
plt.bar(corpus_counts.keys(), corpus_counts.values())
plt.ylabel("Number of LabMT words present")
plt.title("LabMT Word Presence Across Corpora")
plt.xticks(rotation=45)
plt.show()

# ---------------------------------------
# 2) Overlap calculations
# ---------------------------------------

twitter_words = set(df[df["twitter_rank"].notna()]["word"])
google_words = set(df[df["google_rank"].notna()]["word"])
nyt_words = set(df[df["nyt_rank"].notna()]["word"])
lyrics_words = set(df[df["lyrics_rank"].notna()]["word"])

print("\nOverlap counts:")

print("Twitter ∩ NYT:", len(twitter_words & nyt_words))
print("Twitter ∩ GoogleBooks:", len(twitter_words & google_words))
print("Twitter ∩ Lyrics:", len(twitter_words & lyrics_words))
print("NYT ∩ GoogleBooks:", len(nyt_words & google_words))
print("NYT ∩ Lyrics:", len(nyt_words & lyrics_words))
print("GoogleBooks ∩ Lyrics:", len(google_words & lyrics_words))

# Words appearing in ALL FOUR corpora
all_four = twitter_words & google_words & nyt_words & lyrics_words
print("Words present in all four corpora:", len(all_four))

# ---------------------------------------
# 3) Scatterplot Twitter vs NYT rank
# (only for words present in both)
# ---------------------------------------

both_mask = df["twitter_rank"].notna() & df["nyt_rank"].notna()
df_both = df[both_mask]

plt.figure()
plt.scatter(df_both["twitter_rank"], df_both["nyt_rank"])
plt.xlabel("Twitter Rank")
plt.ylabel("NYT Rank")
plt.title("Twitter vs NYT Rank (Shared Words)")
plt.show()

#Overlap matrix between corpora

print_section("Heatmap-like Table: Corpus Overlap Matrix")

# -----------------------------------------------------------------------------
# Step 1: Create boolean flags for each corpus
# -----------------------------------------------------------------------------

# For each word, we record True if it appears in that corpus (has a rank),
# False if it does not appear (rank is missing)

print("\nCreating presence flags for each corpus...")
print("(True = word appears in corpus, False = word does not appear)")

# Create a DataFrame with 4 columns, one for each corpus
# Each column contains True/False values based on whether the word has a rank
flags = pd.DataFrame({
    "Twitter": df["twitter_rank"].notna(),        # True if twitter_rank is not NaN
    "Google Books": df["google_rank"].notna(),    # True if google_rank is not NaN
    "NY Times": df["nyt_rank"].notna(),           # True if nyt_rank is not NaN
    "Lyrics": df["lyrics_rank"].notna()           # True if lyrics_rank is not NaN
})

# Display a small sample to help users understand what the flags look like
print("\nSample of presence flags (first 10 words):")
sample_flags = flags.head(10)                     # Take first 10 rows
sample_flags.index = df["word"].head(10)          # Replace numeric index with actual words
print(sample_flags.to_string())                   # Print the sample

print(f"\nShape of flags matrix: {flags.shape[0]} rows x {flags.shape[1]} columns")
print(f"Each row represents one word, each column represents one corpus")

# -----------------------------------------------------------------------------
# Step 2: Build the overlap matrix
# -----------------------------------------------------------------------------
# The overlap matrix shows:
# - Diagonal: number of words present in each corpus
# - Off-diagonal: number of words present in BOTH corpora

print("\nBuilding overlap matrix...")

# Get the list of corpus names from the flags DataFrame columns
corpus_names = list(flags.columns)
print(f"Corpora being compared: {corpus_names}")

# Create an empty DataFrame to hold the overlap matrix
# Rows and columns will both use the corpus names as labels
overlap_matrix = pd.DataFrame(index=corpus_names, columns=corpus_names)

# Fill in the matrix using nested loops
# The outer loop iterates through rows, inner loop iterates through columns
for i, row_corpus in enumerate(corpus_names):          # i = row index, row_corpus = row name
    for j, col_corpus in enumerate(corpus_names):      # j = column index, col_corpus = column name
        if i == j:
            # Diagonal case (same corpus)
            # Count how many words appear in this corpus
            # flags[row_corpus] gives the True/False column, .sum() counts the Trues
            overlap_matrix.loc[row_corpus, col_corpus] = flags[row_corpus].sum()
        else:
            # Off-diagonal case (different corpora)
            # Count words that appear in BOTH corpora
            # The & operator performs element-wise AND between two boolean Series
            # Result is True only when a word appears in both corpora
            both = flags[row_corpus] & flags[col_corpus]
            overlap_matrix.loc[row_corpus, col_corpus] = both.sum()

# Convert all values to integers for cleaner display
# Word counts should be integers, not floats
overlap_matrix = overlap_matrix.astype(int)

# -----------------------------------------------------------------------------
# Step 3: Display the overlap matrix
# -----------------------------------------------------------------------------

print("\nDisplaying overlap matrix")
print("=" * 70)
print("CORPUS OVERLAP MATRIX")
print("=" * 70)
print("Diagonal: Number of words in each corpus")
print("Off-diagonal: Number of words appearing in BOTH corpora")
print("=" * 70)

# Print the matrix with nice formatting
print("\n" + overlap_matrix.to_string())

print("\nMatrix Interpretation:")
print("- Higher numbers on diagonal mean more words in that corpus")
print("- Higher off-diagonal numbers mean more similar corpora")
print("- Lower off-diagonal numbers mean more different corpora")

# -----------------------------------------------------------------------------
# Step 4: Calculate additional statistics for deeper understanding
# -----------------------------------------------------------------------------

print("\n" + "=" * 70)
print("OVERLAP STATISTICS")
print("=" * 70)

# Calculate total number of words in the dataset
total_words = len(df)
print(f"\nTotal words in labMT dataset: {total_words}")

# Calculate words that appear in ALL four corpora
# Use & operator to combine all four conditions
all_four = (
    flags["Twitter"] & 
    flags["Google Books"] & 
    flags["NY Times"] & 
    flags["Lyrics"]
).sum()
all_four_pct = round(all_four / total_words * 100, 1)
print(f"Words in ALL four corpora: {all_four} ({all_four_pct}%)")
print("  -> These are core vocabulary words used across all contexts")

# Calculate words that appear in NO corpus (missing from all top-5000 lists)
# Use ~ operator to negate (True becomes False, False becomes True)
none_corpus = (
    ~flags["Twitter"] & 
    ~flags["Google Books"] & 
    ~flags["NY Times"] & 
    ~flags["Lyrics"]
).sum()
none_pct = round(none_corpus / total_words * 100, 1)
print(f"Words in NO corpus top-5000: {none_corpus} ({none_pct}%)")
print("  -> These are rare words or specialized vocabulary")

# -----------------------------------------------------------------------------
# Step 5: Create heatmap visualization of the overlap matrix
# -----------------------------------------------------------------------------
# This satisfies the requirement: "Make at least one plot about corpus differences"
# Heatmap is a great way to visualize the overlap matrix

print("\nCreating heatmap visualization...")

# Create a new figure with appropriate size
plt.figure(figsize=(8, 6))

# Create the heatmap using matplotlib's imshow function
# overlap_matrix.values gives the numeric data as a 2D array
# cmap='YlOrRd' means Yellow-Orange-Red color scheme (darker = higher values)
# aspect='auto' ensures the plot fills the figure properly
# vmin=0, vmax=10222 sets the color scale from 0 to total words
heatmap = plt.imshow(overlap_matrix.values, cmap='YlOrRd', aspect='auto', 
                    vmin=0, vmax=len(df))

# Add a color bar to show what colors mean
cbar = plt.colorbar(heatmap)
cbar.set_label('Number of words', fontsize=10)

# Set the ticks and labels
# We want ticks at 0,1,2,3 for our 4 corpora
plt.xticks(range(len(corpus_names)), corpus_names, rotation=45, ha='right')
plt.yticks(range(len(corpus_names)), corpus_names)

# Add title and labels
plt.title('Corpus Overlap Heatmap\n(darker color = more words)', fontsize=12, fontweight='bold')
plt.xlabel('Corpus', fontsize=10)
plt.ylabel('Corpus', fontsize=10)

# Add the actual numbers in each cell
for i in range(len(corpus_names)):
    for j in range(len(corpus_names)):
        # Get the value for this cell
        value = overlap_matrix.iloc[i, j]
        
        # Choose text color based on background darkness
        # If value is high (dark background), use white text; otherwise black
        text_color = 'white' if value > len(df)/2 else 'black'
        
        # Add text to the plot
        plt.text(j, i, f'{value}', ha='center', va='center', 
                color=text_color, fontsize=10, fontweight='bold')

# Adjust layout to prevent cutting off labels
plt.tight_layout()

# Save the figure
save_figure("corpus_overlap_heatmap.png", dpi=300)
print("  - Saved: corpus_overlap_heatmap.png (heatmap visualization)")

# Don't show the plot, just save it
# plt.show()  # Commented out to prevent pop-up window
plt.close()

# -----------------------------------------------------------------------------
# 3. Qualitative exploration: close reading the lexicon as a cultural artifact
# -----------------------------------------------------------------------------

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
        print(f" Loaded data from: {path}")
        print(f"Shape: {df.shape}")
        break
    except FileNotFoundError:
        continue

if df is None:
    print(" Could not find data file. Please check the path.")
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

# Display as regular table
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 20)

print("="*50)
print("REGULAR TABLE OUTPUT")
print("="*50)
print(exhibit.to_string(index=False))

# Save to CSV
exhibit.to_csv('data/word_exhibit.csv', index=False)
print("\n✅ Exhibit saved to data/word_exhibit.csv")

# ============================================
# CREATE MARKDOWN TABLES FOR README
# ============================================
print("\n" + "="*50)
print("MARKDOWN TABLES FOR README (COPY THESE!)")
print("="*50)
print("\n")

# Very Positive Words Table
print("### 📈 Very Positive Words (Happiness > 8.0)")
print("| Word | Score | SD | Corpus Presence |")
print("|------|-------|-----|-----------------|")
for _, row in positive_words.iterrows():
    print(f"| {row['word']} | {row['happiness_average']:.2f} | {row['happiness_standard_deviation']:.2f} | {row['corpus_presence']} |")
print("\n**Interpretative Paragraph:**")
print("[Write 2-3 sentences about what these words reveal about happiness and language]\n")

# Very Negative Words Table
print("### 📉 Very Negative Words (Happiness < 1.5)")
print("| Word | Score | SD | Corpus Presence |")
print("|------|-------|-----|-----------------|")
for _, row in negative_words.iterrows():
    print(f"| {row['word']} | {row['happiness_average']:.2f} | {row['happiness_standard_deviation']:.2f} | {row['corpus_presence']} |")
print("\n**Interpretative Paragraph:**")
print("[Write 2-3 sentences about what these words reveal about negativity and trauma in language]\n")

# Highly Contested Words Table
print("### ⚖️ Highly Contested Words (Highest Standard Deviation)")
print("| Word | Score | SD | Corpus Presence |")
print("|------|-------|-----|-----------------|")
for _, row in contested_words.iterrows():
    print(f"| {row['word']} | {row['happiness_average']:.2f} | {row['happiness_standard_deviation']:.2f} | {row['corpus_presence']} |")
print("\n**Interpretative Paragraph:**")
print("[Write 2-3 sentences about why these words create disagreement and how context matters]\n")

# Culturally Interesting Words Table
print("### 🌍 Culturally/Historically Interesting Words")
print("| Word | Score | SD | Corpus Presence |")
print("|------|-------|-----|-----------------|")
for _, row in interesting_words.iterrows():
    print(f"| {row['word']} | {row['happiness_average']:.2f} | {row['happiness_standard_deviation']:.2f} | {row['corpus_presence']} |")
print("\n**Interpretative Paragraph:**")
print("[Write 2-3 sentences about how these words reflect cultural and linguistic change]\n")

print("="*50)
print("✅ Copy the tables above and paste them into your README.md")
print("="*50)


