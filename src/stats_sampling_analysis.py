import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "met_aesthetic_scored.csv"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"

TABLES_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def load_scored_data():
    print("DATA_PATH =", DATA_PATH)

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    print("shape =", df.shape)
    print("columns =", df.columns.tolist())

    core_cols = [
        "object_id",
        "title",
        "category",
        "term_used",
        "score",
        "coverage",
        "matched",
        "total"
    ]

    missing_cols = [col for col in core_cols if col not in df.columns]
    print("missing_cols =", missing_cols)

    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    df = df[core_cols].copy()

    df["category"] = df["category"].astype(str).str.strip().str.lower()
    df["term_used"] = df["term_used"].astype(str).str.strip().str.lower()
    df["title"] = df["title"].astype(str).str.strip()

    numeric_cols = ["score", "coverage", "matched", "total"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def sample_audit(df):
    category_audit = df.groupby("category").agg(
        total_artworks=("object_id", "count"),
        scored_artworks=("score", lambda x: x.notna().sum()),
        mean_coverage=("coverage", "mean"),
        median_coverage=("coverage", "median"),
        no_match_count=("coverage", lambda x: (x == 0).sum())
    ).reset_index()

    category_audit["scoring_rate"] = (
        category_audit["scored_artworks"] / category_audit["total_artworks"]
    )
    category_audit["no_match_rate"] = (
        category_audit["no_match_count"] / category_audit["total_artworks"]
    )

    category_audit.to_csv(TABLES_DIR / "sample_audit_by_category.csv", index=False)

    term_audit = df.groupby(["category", "term_used"]).agg(
        n=("object_id", "count"),
        scored_n=("score", lambda x: x.notna().sum()),
        mean_score=("score", "mean"),
        median_score=("score", "median"),
        mean_coverage=("coverage", "mean")
    ).reset_index()

    term_audit.to_csv(TABLES_DIR / "sample_audit_by_term.csv", index=False)
    return category_audit, term_audit


def bootstrap_mean_ci(values, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    values = np.array(values.dropna())

    if len(values) == 0:
        raise ValueError("bootstrap_mean_ci received an empty array.")

    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(sample.mean())

    boots = np.array(boots)

    return {
        "mean": values.mean(),
        "ci_lower": np.percentile(boots, 2.5),
        "ci_upper": np.percentile(boots, 97.5),
        "boot_dist": boots
    }


def bootstrap_diff_ci(east, west, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    east = np.array(east.dropna())
    west = np.array(west.dropna())

    if len(east) == 0 or len(west) == 0:
        raise ValueError("bootstrap_diff_ci received an empty group.")

    diffs = []
    for _ in range(n_boot):
        east_sample = rng.choice(east, size=len(east), replace=True)
        west_sample = rng.choice(west, size=len(west), replace=True)
        diffs.append(east_sample.mean() - west_sample.mean())

    diffs = np.array(diffs)

    return {
        "mean_diff": east.mean() - west.mean(),
        "ci_lower": np.percentile(diffs, 2.5),
        "ci_upper": np.percentile(diffs, 97.5),
        "pr_east_gt_west": (diffs > 0).mean(),
        "boot_dist": diffs
    }


def bootstrap_median_diff_ci(east, west, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    east = np.array(east.dropna())
    west = np.array(west.dropna())

    if len(east) == 0 or len(west) == 0:
        raise ValueError("bootstrap_median_diff_ci received an empty group.")

    diffs = []
    for _ in range(n_boot):
        east_sample = rng.choice(east, size=len(east), replace=True)
        west_sample = rng.choice(west, size=len(west), replace=True)
        diffs.append(np.median(east_sample) - np.median(west_sample))

    diffs = np.array(diffs)

    return {
        "median_diff": np.median(east) - np.median(west),
        "ci_lower": np.percentile(diffs, 2.5),
        "ci_upper": np.percentile(diffs, 97.5)
    }


def coverage_sensitivity(df, thresholds=(0.0, 0.3, 0.5)):
    rows = []

    for th in thresholds:
        sub = df[(df["score"].notna()) & (df["coverage"] >= th)].copy()
        east = sub[sub["category"] == "eastern"]["score"]
        west = sub[sub["category"] == "western"]["score"]

        if len(east) == 0 or len(west) == 0:
            continue

        diff_res = bootstrap_diff_ci(east, west)

        rows.append({
            "coverage_threshold": th,
            "n_east": len(east),
            "n_west": len(west),
            "east_mean": east.mean(),
            "west_mean": west.mean(),
            "mean_diff_east_minus_west": diff_res["mean_diff"],
            "ci_lower": diff_res["ci_lower"],
            "ci_upper": diff_res["ci_upper"],
            "pr_east_gt_west": diff_res["pr_east_gt_west"]
        })

    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "coverage_sensitivity.csv", index=False)
    return out


def plot_bootstrap_difference(diff_result):
    diff_boot = diff_result["boot_dist"]
    mean_diff = diff_result["mean_diff"]
    ci_lower = diff_result["ci_lower"]
    ci_upper = diff_result["ci_upper"]

    plt.figure(figsize=(9, 5.5))
    plt.hist(diff_boot, bins=40, edgecolor="white", alpha=0.9)

    plt.axvline(0, linestyle="--", linewidth=2, label="No difference (0)")
    plt.axvline(mean_diff, linewidth=2, label=f"Mean diff = {mean_diff:.3f}")
    plt.axvline(ci_lower, linestyle="--", linewidth=2, label=f"95% CI = [{ci_lower:.3f}, {ci_upper:.3f}]")
    plt.axvline(ci_upper, linestyle="--", linewidth=2)

    plt.title("Bootstrap distribution of mean difference\n(Eastern - Western)")
    plt.xlabel("Mean difference")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "bootstrap_difference_distribution.png", dpi=300)
    plt.close()


def plot_term_sample_sizes(df):
    term_counts = df.groupby(["category", "term_used"]).size().reset_index(name="n")
    term_counts = term_counts.sort_values("n", ascending=False).reset_index(drop=True)

    colors = term_counts["category"].map({
        "eastern": "orange",
        "western": "steelblue"
    })

    plt.figure(figsize=(12, 6))
    plt.bar(term_counts["term_used"], term_counts["n"], color=colors)

    plt.xticks(rotation=75, ha="right")
    plt.title("Sample size by search term")
    plt.xlabel("Search term")
    plt.ylabel("Number of artworks")

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="orange", label="Eastern"),
        Patch(facecolor="steelblue", label="Western")
    ]
    plt.legend(handles=legend_handles)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sample_size_by_term.png", dpi=300)
    plt.close()


def plot_coverage_by_category(df):
    sub = df[df["score"].notna()].copy()

    eastern_cov = sub[sub["category"] == "eastern"]["coverage"].dropna()
    western_cov = sub[sub["category"] == "western"]["coverage"].dropna()

    data = [eastern_cov, western_cov]
    labels = ["Eastern", "Western"]

    plt.figure(figsize=(8, 5.5))
    box = plt.boxplot(data, labels=labels, patch_artist=True)

    box["boxes"][0].set_facecolor("orange")
    box["boxes"][1].set_facecolor("steelblue")

    plt.title("Coverage by category")
    plt.ylabel("Coverage (matched words / total words)")
    plt.xlabel("Category")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "coverage_by_category.png", dpi=300)
    plt.close()


def main():
    df = load_scored_data()

    print("\nRunning sample audit...")
    category_audit, term_audit = sample_audit(df)

    scored = df[df["score"].notna()].copy()
    east = scored[scored["category"] == "eastern"]["score"]
    west = scored[scored["category"] == "western"]["score"]

    print("\nRunning bootstrap group CIs...")
    east_ci = bootstrap_mean_ci(east)
    west_ci = bootstrap_mean_ci(west)

    print("\nRunning bootstrap difference CI...")
    diff_ci = bootstrap_diff_ci(east, west)

    print("\nRunning bootstrap median robustness check...")
    median_diff = bootstrap_median_diff_ci(east, west)

    group_ci_df = pd.DataFrame([
        {
            "group": "eastern",
            "mean": east_ci["mean"],
            "ci_lower": east_ci["ci_lower"],
            "ci_upper": east_ci["ci_upper"]
        },
        {
            "group": "western",
            "mean": west_ci["mean"],
            "ci_lower": west_ci["ci_lower"],
            "ci_upper": west_ci["ci_upper"]
        }
    ])
    group_ci_df.to_csv(TABLES_DIR / "bootstrap_group_cis.csv", index=False)

    diff_summary_df = pd.DataFrame([{
        "estimand": "mean_diff_east_minus_west",
        "estimate": diff_ci["mean_diff"],
        "ci_lower": diff_ci["ci_lower"],
        "ci_upper": diff_ci["ci_upper"],
        "pr_east_gt_west": diff_ci["pr_east_gt_west"],
        "median_diff": median_diff["median_diff"],
        "median_ci_lower": median_diff["ci_lower"],
        "median_ci_upper": median_diff["ci_upper"]
    }])
    diff_summary_df.to_csv(TABLES_DIR / "bootstrap_difference_summary.csv", index=False)

    print("\nRunning coverage sensitivity analysis...")
    sensitivity_df = coverage_sensitivity(df)

    print("\nCreating figures...")
    plot_bootstrap_difference(diff_ci)
    plot_term_sample_sizes(df)
    plot_coverage_by_category(df)

    print("\nDone.")
    print("\nKey outputs created:")

    print(f"  TABLE: {TABLES_DIR / 'sample_audit_by_category.csv'}")
    print(f"  TABLE: {TABLES_DIR / 'sample_audit_by_term.csv'}")
    print(f"  TABLE: {TABLES_DIR / 'bootstrap_group_cis.csv'}")
    print(f"  TABLE: {TABLES_DIR / 'bootstrap_difference_summary.csv'}")
    print(f"  TABLE: {TABLES_DIR / 'coverage_sensitivity.csv'}")

    print(f"  FIGURE: {FIGURES_DIR / 'bootstrap_difference_distribution.png'}")
    print(f"  FIGURE: {FIGURES_DIR / 'sample_size_by_term.png'}")
    print(f"  FIGURE: {FIGURES_DIR / 'coverage_by_category.png'}")

    print("\nMain inferential results:")
    print(f"  Eastern mean = {east_ci['mean']:.3f} [{east_ci['ci_lower']:.3f}, {east_ci['ci_upper']:.3f}]")
    print(f"  Western mean = {west_ci['mean']:.3f} [{west_ci['ci_lower']:.3f}, {west_ci['ci_upper']:.3f}]")
    print(f"  Mean diff (East - West) = {diff_ci['mean_diff']:.3f} [{diff_ci['ci_lower']:.3f}, {diff_ci['ci_upper']:.3f}]")
    print(f"  Pr(East > West) = {diff_ci['pr_east_gt_west']:.3f}")
    print(f"  Median diff (East - West) = {median_diff['median_diff']:.3f} [{median_diff['ci_lower']:.3f}, {median_diff['ci_upper']:.3f}]")

    print("\nCoverage sensitivity summary:")
    print(sensitivity_df.to_string(index=False))


if __name__ == "__main__":
    main()