from pathlib import Path
import pandas as pd

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

# ===========================
# 1.1 Load the tab-delimited labMT dataset
# ===========================
# Read the tab-delimited file, skipping metadata
df = pd.read_csv(
    "data/raw/Data_Set_S1.txt",
    sep="\t",        # <--- file is tab-separated
    skiprows=3,       # <--- skip first 3 lines (metadata)
    na_values="--"    # <--- treat '--' as missing
)

# Show the first 5 rows to check if loaded correctly
print(df.head())  # <--- comment: preview the first few rows

# Convert numeric columns to numeric types explicitly
numeric_cols = ["happiness_rank", "happiness_average", 
                "happiness_standard_deviation", "twitter_rank", 
                "google_rank", "nyt_rank", "lyrics_rank"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")  
    # comment: coerce invalid parsing to NaN

# Check the data types after conversion
print(df.dtypes)  # <--- comment: ensure numeric columns are numeric

# Confirm the number of rows and columns
print("Dataset shape:", df.shape)

# ===========================
# 1.2 Create a Data Dictionary
# ===========================

# List each column and its data type
print("Column names and data types:")
print(df.dtypes)  # <--- comment: display all column names with types

# Count missing values per column
missing_counts = df.isna().sum()
print("\nMissing values per column:")
print(missing_counts)  # <--- comment: count how many NaNs in each column

# Optional: create a structured data dictionary as a DataFrame for easier display
data_dict = pd.DataFrame({
    "Column": df.columns,
    "Type": [str(df[col].dtype) for col in df.columns],
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

# -------------------------------
# 1.3 Sanity Checks
# -------------------------------


# Check for duplicated words
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