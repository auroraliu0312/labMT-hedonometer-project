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

cols_to_show = ['word', 'happiness_rank', 'happiness_average', 
                'happiness_standard_deviation', 'twitter_rank',
                'google_rank', 'nyt_rank', 'lyrics_rank']

sample_table = sample_rows[cols_to_show].copy()


sample_table.to_csv('tables/random_sample_15_rows.csv', index=False)


for idx, row in sample_table.iterrows():
    google = f"{row['google_rank']:.1f}" if pd.notna(row['google_rank']) else "NaN"
    nyt = f"{row['nyt_rank']:.1f}" if pd.notna(row['nyt_rank']) else "NaN"
    lyrics = f"{row['lyrics_rank']:.1f}" if pd.notna(row['lyrics_rank']) else "NaN"
    
    print(f"| {row['word']} | {int(row['happiness_rank'])} | {row['happiness_average']:.2f} | {google} | {nyt} | {lyrics} |")

# Identify the 10 most positive words
top_10_positive = df.nlargest(10, "happiness_average")[["word", "happiness_average"]]  # English comment: Top 10 highest happiness scores
print("Top 10 positive words:\n", top_10_positive)

# Identify the 10 most negative words
bottom_10_negative = df.nsmallest(10, "happiness_average")[["word", "happiness_average"]]  # English comment: Top 10 lowest happiness scores
print("Top 10 negative words:\n", bottom_10_negative)


# -----------------------------------------------------------------------------
# 2.1 Distribution of happiness_average (Single Enhanced Histogram)
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
            float(h.shape[0]),
            float(h.mean()),
            float(h.median()),
            float(h.std()),
            float(h.quantile(0.05)),
            float(h.quantile(0.95)),
        ],
    }
)

print(summary_stats.to_string(index=False))
save_csv(summary_stats, "happiness_average_summary_stats.csv", index=False)

# -----------------------------------------------------------------------------
# Find the actual words for extremes
# -----------------------------------------------------------------------------
most_negative_word = df.loc[df['happiness_average'].idxmin(), 'word']
most_positive_word = df.loc[df['happiness_average'].idxmax(), 'word']

print(f"Most negative word: '{most_negative_word}' with score {h.min():.2f}")
print(f"Most positive word: '{most_positive_word}' with score {h.max():.2f}")

# -----------------------------------------------------------------------------
# SINGLE ENHANCED HISTOGRAM with all annotations
# -----------------------------------------------------------------------------
plt.figure(figsize=(14, 10))

# Calculate key values
p05 = h.quantile(0.05)      # 3.18
p95 = h.quantile(0.95)      # 7.08
min_val = h.min()            # 1.30
max_val = h.max()            # 8.50
mean_val = h.mean()          # 5.38
median_val = h.median()      # 5.44

# Plot histogram
n, bins, patches = plt.hist(h, bins=40, color='skyblue', edgecolor='black', alpha=0.7)

# Color the tails differently to highlight them
for i, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
    if left < p05:  # Negative tail
        patches[i].set_facecolor('lightcoral')
        patches[i].set_alpha(0.8)
    elif right > p95:  # Positive tail
        patches[i].set_facecolor('lightgreen')
        patches[i].set_alpha(0.8)

# Add vertical lines for key percentiles
plt.axvline(p05, color='red', linestyle='--', linewidth=2.5, alpha=0.8,
            label=f'5th percentile: {p05:.2f} (start of negative tail)')
plt.axvline(p95, color='green', linestyle='--', linewidth=2.5, alpha=0.8,
            label=f'95th percentile: {p95:.2f} (start of positive tail)')

# Add vertical lines for extreme values
plt.axvline(min_val, color='darkred', linestyle=':', linewidth=2.5, alpha=0.8,
            label=f'Minimum: {min_val:.2f} (most negative word)')
plt.axvline(max_val, color='darkgreen', linestyle=':', linewidth=2.5, alpha=0.8,
            label=f'Maximum: {max_val:.2f} (most positive word)')

# Add mean and median for reference
plt.axvline(mean_val, color='blue', linestyle='-', linewidth=2, alpha=0.6,
            label=f'Mean: {mean_val:.2f}')
plt.axvline(median_val, color='purple', linestyle='-', linewidth=2, alpha=0.6,
            label=f'Median: {median_val:.2f}')

# Add shaded regions for tails
plt.axvspan(1, p05, alpha=0.1, color='red')
plt.axvspan(p95, 9, alpha=0.1, color='green')

# Add annotations for the words
max_count = max(n)

# Annotate most negative word
plt.annotate(f'Most negative: "{most_negative_word}"\nScore: {min_val:.2f}',
             xy=(min_val, 5), xytext=(min_val - 0.8, max_count * 0.8),
             arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
             fontsize=11, color='darkred',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

# Annotate most positive word
plt.annotate(f'Most positive: "{most_positive_word}"\nScore: {max_val:.2f}',
             xy=(max_val, 5), xytext=(max_val + 0.5, max_count * 0.7),
             arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
             fontsize=11, color='darkgreen',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

# Add text box with your exact finding
finding_text = (
    f"KEY FINDING:\n"
    f"• Negative tail width: 1 to {p05:.2f} = {p05-1:.2f} points\n"
    f"• Positive tail width: {p95:.2f} to 9 = {9-p95:.2f} points\n"
    f"• Negative tail is {(p05-1) - (9-p95):.2f} points LONGER\n"
    f"  → More mildly negative words in vocabulary\n\n"
    f"• Most negative word '{most_negative_word}': {min_val:.2f}\n"
    f"  Distance from mean: {mean_val-min_val:.2f}\n"
    f"• Most positive word '{most_positive_word}': {max_val:.2f}\n"
    f"  Distance from mean: {max_val-mean_val:.2f}\n"
    f"• Positive extreme is {(max_val-mean_val) - (mean_val-min_val):.2f} points FURTHER\n"
    f"  → Most positive words are more extremely positive"
)

plt.text(0.98, 0.98, finding_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.95),
         linespacing=1.5)

# Title and labels
plt.title('Figure 1: Distribution of Happiness Scores\nwith Highlighted Tails and Extremes', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Happiness Score (1-9)', fontsize=13, labelpad=10)
plt.ylabel('Number of Words', fontsize=13, labelpad=10)
plt.legend(loc='upper left', fontsize=9, framealpha=0.95)
plt.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
save_figure("happiness_distribution_enhanced.png")
plt.close()

print_section("2.2 Disagreement: happiness_standard_deviation")

print_section("2.2 Disagreement: happiness_standard_deviation")

# ---------------------------
# 2.2 Disagreement: which words are “contested”?
# ---------------------------
print("\n--- Disagreement analysis (average vs standard deviation) ---")

# Scatterplot:
# - x = average happiness (valence)
# - y = standard deviation (how much raters disagreed)
# Each dot = one word in the dataset
plt.figure()
plt.scatter(
        df["happiness_average"],
        df["happiness_standard_deviation"],
    )
plt.xlabel("Happiness Average")
plt.ylabel("Happiness Standard Deviation")
plt.title("Average Happiness vs Disagreement (LabMT)")
plt.close()

# Identify the 15 most disagreed-about words (highest std dev)
most_disagreed = df.sort_values("happiness_standard_deviation", ascending=False).head(15)

print("\nTop 15 most disagreed-about words (highest std dev):")
print(most_disagreed[["word", "happiness_average", "happiness_standard_deviation"]])

# ---------------------------
# 2.3 Corpus comparison
# ---------------------------
print("\n--- Corpus comparison ---")

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
plt.close()

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
plt.close()

# Heatmap-like table - Overlap matrix between corpora

print_section("Heatmap-like Table: Corpus Overlap Matrix")

# -----------------------------------------------------------------------------
# Step 1: Create boolean flags for each corpus
# -----------------------------------------------------------------------------
# For each word, we record True if it appears in that corpus (has a rank),
# False if it does not appear (rank is missing)

print("\nStep 1: Creating presence flags for each corpus...")
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

print("\nStep 2: Building overlap matrix...")

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

print("\nStep 3: Displaying overlap matrix")
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

print("\nStep 3.5: Creating heatmap visualization...")

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

# ---------------------------
# 3.1 Build a small “exhibit” of words (20 words)
# ---------------------------
print("\n--- 3.1 Exhibit of words (20-word table) ---")

    # 5 very positive / 5 very negative
very_positive = (
        df.sort_values("happiness_average", ascending=False)
        .head(5)
        .assign(category="very_positive")
    )

very_negative = (
        df.sort_values("happiness_average", ascending=True)
        .head(5)
        .assign(category="very_negative")
    )

    # 5 highly contested (highest std dev)
    # (Optional: avoid overlap with positive/negative by filtering them out)
exclude = set(very_positive["word"]) | set(very_negative["word"])

highly_contested = (
        df[~df["word"].isin(exclude)]
        .sort_values("happiness_standard_deviation", ascending=False)
        .head(5)
        .assign(category="highly_contested")
    )

    # 5 “weird" or "culturally loaded" words

weird_words = [
        "capitalism",
        "churches",
        "whiskey",
        "porn",
        "weekend"
    ]

weird = (
        df[df["word"].isin(weird_words)]
        .copy()
        .assign(category="weird_or_culturally_loaded")
    )


if len(weird) < 5:
        missing = [w for w in weird_words if w not in set(df["word"])]
        print("WARNING: These weird_words were not found in the dataset:", missing)

# Combine into one exhibit table
exhibit = pd.concat([very_positive, very_negative, highly_contested, weird], ignore_index=True)

# Keep the most relevant columns for the exhibit
exhibit = exhibit[["category", "word", "happiness_average", "happiness_standard_deviation",
                    "twitter_rank", "google_rank", "nyt_rank", "lyrics_rank"]]

print("\nExhibit table (preview):")
print(exhibit)

# Save to tables/
from pathlib import Path
TABLES_DIR = Path("tables")
TABLES_DIR.mkdir(parents=True, exist_ok=True)
out_path = TABLES_DIR / "exhibit_words.csv"
exhibit.to_csv(out_path, index=False)
print(f"\nSaved exhibit table to: {out_path}")

# -----------------------------------------------------------------------------
# Save labMT word list for future use
# -----------------------------------------------------------------------------
print("\nSaving labMT word list for future use...")

# Create a clean version with just word and happiness score
labMT_clean = df[['word', 'happiness_average']].copy()

# Save to processed folder
labMT_clean_path = Path("data/processed") / "labMT_cleaned.csv"
labMT_clean.to_csv(labMT_clean_path, index=False)
print(f"Saved labMT word list to: {labMT_clean_path}")

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
print_section("Done")