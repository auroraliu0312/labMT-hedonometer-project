from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from load_data import load_dataset, DATA_PATH
def main() -> None:
    # ---------------------------
    # Load dataset
    # ---------------------------
    # Load the cleaned DataFrame produced by load_data.py (handles header lines, "--" -> NaN, numeric conversion)
    df = load_dataset(DATA_PATH)

    # ---------------------------
    # 1.3 Sanity checks
    # ---------------------------
    print("\n--- Duplicate check ---")
    # duplicated() marks True for repeated values after the first occurrence
    duplicate_words = df["word"].duplicated().sum()
    print("Number of duplicated words:", duplicate_words)

    # If duplicates exist, print all duplicated rows (keep=False keeps all occurrences)
    if duplicate_words > 0:
        print("Duplicated entries (all occurrences):")
        dup_df = df[df["word"].duplicated(keep=False)].sort_values("word")
        print(dup_df)

    print("\n--- Random sample (15 rows) ---")
    # random_state ensures you and your teammates can reproduce the same sample
    print(df.sample(15, random_state=42))

    print("\n--- Top 10 most positive words ---")
    top_positive = df.sort_values("happiness_average", ascending=False).head(10)
    print(top_positive[["word", "happiness_average"]])

    print("\n--- Top 10 most negative words ---")
    top_negative = df.sort_values("happiness_average", ascending=True).head(10)
    print(top_negative[["word", "happiness_average"]])

    # ---------------------------
    # 2.1 Distribution of happiness scores
    # ---------------------------
    print("\n--- Distribution of happiness_average ---")

    # Histogram: shows how happiness scores are distributed across all words
    plt.figure()
    plt.hist(df["happiness_average"], bins=30)
    plt.xlabel("Happiness Average")
    plt.ylabel("Frequency")
    plt.title("Distribution of Happiness Scores (LabMT)")
    plt.show()

    # Summary statistics: central tendency + spread + tails
    mean_val = df["happiness_average"].mean()
    median_val = df["happiness_average"].median()
    std_val = df["happiness_average"].std()
    p5 = df["happiness_average"].quantile(0.05)
    p95 = df["happiness_average"].quantile(0.95)

    print("\nSummary statistics for happiness_average:")
    print("Mean:", mean_val)
    print("Median:", median_val)
    print("Standard deviation:", std_val)
    print("5th percentile:", p5)
    print("95th percentile:", p95)

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
    plt.show()

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
    # 3) Optional: Scatterplot Twitter vs NYT rank
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

    # 5 “weird / surprising / historically dated / culturally loaded” (your choice)
    # IMPORTANT: you must edit this list yourself (this is the humanities part).
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

    # If some of your weird words were not found, you'll get <5 rows here.
    # That’s fine: fix by choosing words that actually exist in df["word"].
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

if __name__ == "__main__":
    main()