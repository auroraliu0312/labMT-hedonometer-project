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

    required_cols = [
        "object_id",
        "title",
        "category",
        "object_begin",
        "score",
        "coverage",
        "matched",
        "total"
    ]

    missing_required = [col for col in required_cols if col not in df.columns]
    print("missing_required =", missing_required)

    if missing_required:
        raise ValueError(f"Missing required columns in dataset: {missing_required}")

    use_cols = required_cols.copy()
    if "term_used" in df.columns:
        use_cols.append("term_used")

    df = df[use_cols].copy()

    df["category"] = df["category"].astype(str).str.strip().str.lower()
    df["title"] = df["title"].astype(str).str.strip()

    if "term_used" in df.columns:
        df["term_used"] = df["term_used"].astype(str).str.strip().str.lower()

    numeric_cols = ["object_begin", "score", "coverage", "matched", "total"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def sample_audit(df):
    category_audit = df.groupby("category").agg(
        total_artworks=("object_id", "count"),
        unique_object_ids=("object_id", "nunique"),
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
    category_audit["duplicate_rows"] = (
        category_audit["total_artworks"] - category_audit["unique_object_ids"]
    )

    category_audit.to_csv(TABLES_DIR / "sample_audit_by_category.csv", index=False)

    if "term_used" in df.columns:
        term_audit = df.groupby(["category", "term_used"]).agg(
            n=("object_id", "count"),
            scored_n=("score", lambda x: x.notna().sum()),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            mean_coverage=("coverage", "mean")
        ).reset_index()

        term_audit.to_csv(TABLES_DIR / "sample_audit_by_term.csv", index=False)
    else:
        term_audit = pd.DataFrame()

    return category_audit, term_audit


def bootstrap_mean_ci(values, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    values = np.array(pd.Series(values).dropna())

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
    east = np.array(pd.Series(east).dropna())
    west = np.array(pd.Series(west).dropna())

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
    east = np.array(pd.Series(east).dropna())
    west = np.array(pd.Series(west).dropna())

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


def assign_period_1800(year):
    if pd.isna(year):
        return np.nan
    elif year < 1800:
        return "Pre-1800"
    else:
        return "Post-1800"

def temporal_coverage_analysis_1800(df):
    """
    Temporal lexical coverage analysis using 1800 as cutoff.
    Uses coverage (matched / total), not happiness score.
    CI follows your classmate's logic: SEM * 1.96
    """
    sub = df.copy()
    sub["period_1800"] = sub["object_begin"].apply(assign_period_1800)
    sub = sub[sub["period_1800"].notna()].copy()

    rows = []

    for period in ["Pre-1800", "Post-1800"]:
        period_df = sub[sub["period_1800"] == period]

        east = period_df[period_df["category"] == "eastern"]["coverage"].dropna()
        west = period_df[period_df["category"] == "western"]["coverage"].dropna()

        if len(east) == 0 or len(west) == 0:
            continue

        east_mean = east.mean()
        west_mean = west.mean()

        east_sem = east.sem()
        west_sem = west.sem()

        east_ci = east_sem * 1.96 if pd.notna(east_sem) else np.nan
        west_ci = west_sem * 1.96 if pd.notna(west_sem) else np.nan

        diff = east_mean - west_mean
        diff_se = np.sqrt((east_sem ** 2) + (west_sem ** 2))
        diff_ci = diff_se * 1.96 if pd.notna(diff_se) else np.nan

        rows.append({
            "period": period,
            "n_east": len(east),
            "n_west": len(west),
            "east_mean": east_mean,
            "east_ci": east_ci,
            "west_mean": west_mean,
            "west_ci": west_ci,
            "diff_east_minus_west": diff,
            "diff_ci": diff_ci
        })

    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "temporal_coverage_analysis_1800.csv", index=False)
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
    plt.axvline(ci_lower, linestyle="--", linewidth=2,
                label=f"95% CI = [{ci_lower:.3f}, {ci_upper:.3f}]")
    plt.axvline(ci_upper, linestyle="--", linewidth=2)

    plt.title("Bootstrap distribution of mean difference\n(Eastern - Western)")
    plt.xlabel("Mean difference")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "bootstrap_difference_distribution.png", dpi=300)
    plt.close()


def plot_term_sample_sizes(df):
    if "term_used" not in df.columns:
        print("Skipping term sample size plot: 'term_used' not found.")
        return

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


def plot_temporal_coverage_comparison_1800(temporal_df):
    """
    Lexical coverage by time period (1800 cutoff) and category.
    Mimics your classmate's figure structure:
    - left axis: mean coverage
    - right axis: East-West coverage difference
    """
    if temporal_df.empty or len(temporal_df) < 2:
        print("Skipping temporal coverage plot: insufficient data.")
        return

    temporal_df = temporal_df.copy()
    temporal_df = temporal_df.set_index("period").loc[["Pre-1800", "Post-1800"]].reset_index()

    periods = temporal_df["period"].tolist()
    x_positions = np.arange(len(periods))
    offset = 0.18

    east_color = "#F27F7F"
    west_color = "#6FD0C8"
    trend_color = "#2C3E50"

    east_means = temporal_df["east_mean"].values
    east_cis = temporal_df["east_ci"].values
    east_ns = temporal_df["n_east"].values

    west_means = temporal_df["west_mean"].values
    west_cis = temporal_df["west_ci"].values
    west_ns = temporal_df["n_west"].values

    differences = temporal_df["diff_east_minus_west"].values

    fig, ax1 = plt.subplots(figsize=(14, 10))

    # Eastern bars
    ax1.bar(
        x_positions - offset,
        east_means,
        width=0.35,
        color=east_color,
        alpha=0.95,
        edgecolor="black",
        linewidth=1.5,
        label="Eastern",
        yerr=east_cis,
        capsize=8,
        error_kw={"linewidth": 2, "ecolor": "black", "capthick": 2}
    )

    # Western bars
    ax1.bar(
        x_positions + offset,
        west_means,
        width=0.35,
        color=west_color,
        alpha=0.95,
        edgecolor="black",
        linewidth=1.5,
        label="Western",
        yerr=west_cis,
        capsize=8,
        error_kw={"linewidth": 2, "ecolor": "black", "capthick": 2}
    )

    # sample sizes
    for i in range(len(periods)):
        ax1.text(
            x_positions[i] - offset,
            east_means[i] + east_cis[i] + 0.035,
            f"n={east_ns[i]}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=east_color
        )
        ax1.text(
            x_positions[i] + offset,
            west_means[i] + west_cis[i] + 0.035,
            f"n={west_ns[i]}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=west_color
        )

    # second axis for difference
    ax2 = ax1.twinx()

    ax2.plot(
        x_positions,
        differences,
        marker="o",
        linewidth=3,
        markersize=12,
        color=trend_color,
        linestyle="-",
        markerfacecolor="white",
        markeredgewidth=3,
        label="Coverage Difference"
    )

    for i, diff in enumerate(differences):
        ax2.text(
            x_positions[i],
            diff + 0.02 if diff >= 0 else diff - 0.04,
            f"{diff:+.3f}",
            ha="center",
            va="bottom" if diff >= 0 else "top",
            fontsize=12,
            fontweight="bold",
            color=trend_color,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="#E8E8E8",
                alpha=0.95,
                edgecolor=trend_color
            )
        )

    # left axis
    ax1.set_ylabel("Mean Lexical Coverage", fontsize=16, fontweight="bold")
    ax1.set_ylim(0.0, 0.85)
    ax1.set_yticks(np.arange(0.0, 0.81, 0.1))

    # 50% coverage reference line
    ax1.axhline(y=0.5, color="gray", linestyle=":", linewidth=1.5, alpha=0.8)

    # right axis
    ax2.set_ylabel("East-West Coverage Difference", fontsize=16, fontweight="bold", color=trend_color)
    ax2.tick_params(axis="y", labelcolor=trend_color, labelsize=12)

    max_abs_diff = max(abs(np.nanmin(differences)), abs(np.nanmax(differences))) + 0.05
    ax2.set_ylim(-max_abs_diff * 2.5, max_abs_diff * 2.5)

    # show right axis as percentage-ish labels
    ticks = ax2.get_yticks()
    ax2.set_yticklabels([f"{int(t*100)}%" if t != 0 else "0%" for t in ticks])

    # x axis
    ax1.set_xlabel("Historical Period", fontsize=16, fontweight="bold")
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(["Pre-1800", "Post-1800"], fontsize=13)

    # title
    ax1.set_title(
        "Lexical Coverage by Time Period (1800 Cutoff) and Category\n"
        "(Error bars = 95% Confidence Intervals)",
        fontsize=20,
        fontweight="bold",
        pad=22
    )

    # legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=east_color, edgecolor="black", label="Eastern"),
        Patch(facecolor=west_color, edgecolor="black", label="Western"),
        Line2D([0], [0], color="gray", linestyle=":", linewidth=1.5, label="50% Coverage"),
        Line2D([0], [0], color=trend_color, marker="o", markersize=10,
               markerfacecolor="white", markeredgewidth=2.5, linewidth=3,
               label="Coverage Difference"),
    ]
    ax1.legend(handles=legend_handles, loc="upper left", fontsize=11,
               framealpha=0.95, edgecolor="black")

    # grid
    ax1.grid(True, axis="y", alpha=0.2)

    # summary box
    data_text = (
        "COVERAGE ANALYSIS SUMMARY (1800 Cutoff):\n"
        f"Pre-1800:  Eastern {east_means[0]*100:.1f}% ± {east_cis[0]*100:.1f}% (n={east_ns[0]})  |  "
        f"Western {west_means[0]*100:.1f}% ± {west_cis[0]*100:.1f}% (n={west_ns[0]})\n"
        f"Post-1800: Eastern {east_means[1]*100:.1f}% ± {east_cis[1]*100:.1f}% (n={east_ns[1]})  |  "
        f"Western {west_means[1]*100:.1f}% ± {west_cis[1]*100:.1f}% (n={west_ns[1]})\n\n"
        f"Difference (East-West): Pre-1800 {differences[0]*100:+.1f}%  |  Post-1800 {differences[1]*100:+.1f}%"
    )

    plt.figtext(
        0.16,
        0.03,
        data_text,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="#F0F0F0", alpha=0.95, edgecolor="gray"),
        family="monospace"
    )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "lexical_coverage_by_time_period_1800.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_temporal_waterfall_1800(temporal_df):
    if temporal_df.empty or len(temporal_df) < 2:
        print("Skipping temporal waterfall plot: insufficient data.")
        return

    temporal_df = temporal_df.copy()
    temporal_df = temporal_df.set_index("period").loc[["Pre-1800", "Post-1800"]].reset_index()

    periods = temporal_df["period"].tolist()
    x_pos = np.arange(len(periods))
    width = 0.6

    east_scores = temporal_df["east_mean"].values
    west_scores = temporal_df["west_mean"].values
    differences = temporal_df["diff_east_minus_west"].values

    east_ci = temporal_df["east_ci"].values
    west_ci = temporal_df["west_ci"].values
    diff_ci = temporal_df["diff_ci"].values

    east_n = temporal_df["n_east"].values
    west_n = temporal_df["n_west"].values

    plt.figure(figsize=(12, 8))

    east_color = "#FF6B6B"
    west_color = "#4ECDC4"
    trend_color = "#2C3E50"

    bars = plt.bar(
        x_pos,
        differences,
        width,
        color=east_color,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
        yerr=diff_ci,
        capsize=8,
        error_kw={"linewidth": 2, "ecolor": "black", "capthick": 2}
    )

    plt.plot(
        x_pos,
        differences,
        "o-",
        color=trend_color,
        linewidth=3,
        markersize=12,
        markerfacecolor="white",
        markeredgewidth=2,
        markeredgecolor=trend_color,
        label="Trend Line"
    )

    x_smooth = np.linspace(0, 1, 50)
    ci_lower = differences - diff_ci
    ci_upper = differences + diff_ci

    ci_lower_smooth = np.interp(x_smooth, [0, 1], ci_lower)
    ci_upper_smooth = np.interp(x_smooth, [0, 1], ci_upper)

    plt.fill_between(
        x_smooth,
        ci_lower_smooth,
        ci_upper_smooth,
        color=trend_color,
        alpha=0.15,
        label="95% CI Region"
    )

    for i, (bar, diff, ci, east, west, east_ci_val, west_ci_val) in enumerate(
        zip(bars, differences, diff_ci, east_scores, west_scores, east_ci, west_ci)
    ):
        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + ci + 0.1,
            f"E-W: {diff:+.2f} ± {ci:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=trend_color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor=trend_color)
        )

        plt.text(
            bar.get_x() + bar.get_width() / 2.0 - 0.15,
            height + 0.3,
            f"E: {east:.2f} ± {east_ci_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=east_color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor=east_color)
        )

        west_y = -0.2 if diff > 0 else height - 0.3
        plt.text(
            bar.get_x() + bar.get_width() / 2.0 + 0.15,
            west_y,
            f"W: {west:.2f} ± {west_ci_val:.2f}",
            ha="center",
            va="top" if diff > 0 else "bottom",
            fontsize=10,
            color=west_color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor=west_color)
        )

        plt.text(
            bar.get_x() + bar.get_width() / 2.0 - 0.15,
            -0.5,
            f"n={east_n[i]}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=east_color,
            fontweight="bold"
        )
        plt.text(
            bar.get_x() + bar.get_width() / 2.0 + 0.15,
            -0.5,
            f"n={west_n[i]}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=west_color,
            fontweight="bold"
        )

        if i > 0:
            prev = differences[i - 1]
            curr = differences[i]

            if prev != 0:
                pct_increase = ((curr - prev) / abs(prev)) * 100
                ci_overlap = (curr - diff_ci[i]) < (prev + diff_ci[i - 1])
                sig_symbol = "✓" if not ci_overlap else "!"
                sig_color = "green" if not ci_overlap else "orange"

                plt.annotate(
                    f"↑ {pct_increase:.0f}%  {sig_symbol}",
                    xy=(x_pos[i - 1] + 0.3, (prev + curr) / 2),
                    xytext=(x_pos[i - 1] + 0.4, (prev + curr) / 2 + 0.3),
                    arrowprops=dict(arrowstyle="->", color=trend_color, lw=2),
                    fontsize=11,
                    fontweight="bold",
                    color=sig_color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor=sig_color)
                )

    for i, (x, east, west, east_ci_val, west_ci_val) in enumerate(
        zip(x_pos, east_scores, west_scores, east_ci, west_ci)
    ):
        plt.hlines(y=east, xmin=x - 0.3, xmax=x - 0.1, colors=east_color, linewidth=2, alpha=0.5)
        plt.hlines(y=west, xmin=x + 0.1, xmax=x + 0.3, colors=west_color, linewidth=2, alpha=0.5)

        plt.errorbar(x - 0.2, east, yerr=east_ci_val, fmt="none", ecolor=east_color, alpha=0.5, capsize=3)
        plt.errorbar(x + 0.2, west, yerr=west_ci_val, fmt="none", ecolor=west_color, alpha=0.5, capsize=3)

    plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.3)

    plt.ylabel("East-West Difference (Eastern - Western)", fontsize=14, fontweight="bold")
    plt.xlabel("Historical Period", fontsize=14, fontweight="bold")
    plt.title(
        "EAST-WEST DIFFERENCE IN HAPPINESS SCORES\n(1800 Cutoff) With 95% Confidence Intervals",
        fontsize=18,
        fontweight="bold",
        pad=20
    )

    plt.xticks(x_pos, periods, fontsize=14)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.2, axis="y")
    plt.ylim(-1.0, 3.0)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=east_color, alpha=0.8, label="East-West Difference"),
        plt.Line2D([0], [0], color=trend_color, linewidth=3, label="Trend Line"),
        Patch(facecolor=trend_color, alpha=0.15, label="95% CI Region"),
        Patch(facecolor=east_color, alpha=0.5, label="Eastern Mean (background)"),
        Patch(facecolor=west_color, alpha=0.5, label="Western Mean (background)")
    ]
    plt.legend(handles=legend_elements, loc="upper left", fontsize=10, framealpha=0.9)

    prev = differences[0]
    curr = differences[1]
    pct = ((curr - prev) / abs(prev)) * 100 if prev != 0 else np.nan
    ci_overlap = (curr - diff_ci[1]) < (prev + diff_ci[0])
    sig_text = "✓ Significant increase (CIs do not overlap)" if not ci_overlap else "! Caution: Confidence intervals overlap"

    data_text = (
        f"DATA SUMMARY (1800 cutoff):\n"
        f"Pre-1800:  Eastern {east_scores[0]:.2f}±{east_ci[0]:.2f} (n={east_n[0]})  |  "
        f"Western {west_scores[0]:.2f}±{west_ci[0]:.2f} (n={west_n[0]})\n"
        f"Post-1800: Eastern {east_scores[1]:.2f}±{east_ci[1]:.2f} (n={east_n[1]})  |  "
        f"Western {west_scores[1]:.2f}±{west_ci[1]:.2f} (n={west_n[1]})\n"
        f"Difference: Pre-1800 {differences[0]:+.2f}±{diff_ci[0]:.2f}  |  "
        f"Post-1800 {differences[1]:+.2f}±{diff_ci[1]:.2f}\n"
        f"Change: {pct:.0f}%  |  {sig_text}"
    )

    plt.figtext(
        0.15,
        0.01,
        data_text,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="#F0F0F0", alpha=0.9, edgecolor="gray"),
        family="monospace"
    )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "east_west_difference_1800_cutoff.png", dpi=300, bbox_inches="tight")
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

    print("\nRunning temporal lexical coverage analysis with 1800 cutoff...")
    temporal_df = temporal_coverage_analysis_1800(df)

    print("\nCreating figures...")
    plot_bootstrap_difference(diff_ci)
    plot_term_sample_sizes(df)
    plot_coverage_by_category(df)
    plot_temporal_coverage_comparison_1800(temporal_df)

    print("\nDone.")
    print("\nKey outputs created:")

    print(f"  TABLE: {TABLES_DIR / 'sample_audit_by_category.csv'}")
    if "term_used" in df.columns:
        print(f"  TABLE: {TABLES_DIR / 'sample_audit_by_term.csv'}")
    print(f"  TABLE: {TABLES_DIR / 'bootstrap_group_cis.csv'}")
    print(f"  TABLE: {TABLES_DIR / 'bootstrap_difference_summary.csv'}")
    print(f"  TABLE: {TABLES_DIR / 'coverage_sensitivity.csv'}")
    print(f"  TABLE: {TABLES_DIR / 'temporal_coverage_analysis_1800.csv'}")

    print(f"  FIGURE: {FIGURES_DIR / 'bootstrap_difference_distribution.png'}")
    if "term_used" in df.columns:
        print(f"  FIGURE: {FIGURES_DIR / 'sample_size_by_term.png'}")
    print(f"  FIGURE: {FIGURES_DIR / 'coverage_by_category.png'}")
    print(f"  FIGURE: {FIGURES_DIR / 'lexical_coverage_by_time_period_1800.png'}")
    print(f"  FIGURE: {FIGURES_DIR / 'east_west_difference_1800_cutoff.png'}")

    print("\nMain inferential results:")
    print(f"  Eastern mean = {east_ci['mean']:.3f} [{east_ci['ci_lower']:.3f}, {east_ci['ci_upper']:.3f}]")
    print(f"  Western mean = {west_ci['mean']:.3f} [{west_ci['ci_lower']:.3f}, {west_ci['ci_upper']:.3f}]")
    print(f"  Mean diff (East - West) = {diff_ci['mean_diff']:.3f} [{diff_ci['ci_lower']:.3f}, {diff_ci['ci_upper']:.3f}]")
    print(f"  Pr(East > West) = {diff_ci['pr_east_gt_west']:.3f}")
    print(f"  Median diff (East - West) = {median_diff['median_diff']:.3f} [{median_diff['ci_lower']:.3f}, {median_diff['ci_upper']:.3f}]")

    print("\nCategory audit summary:")
    print(category_audit.to_string(index=False))

    print("\nCoverage sensitivity summary:")
    print(sensitivity_df.to_string(index=False))

    if not temporal_df.empty:
        print("\nTemporal lexical coverage analysis (1800 cutoff):")
        print(temporal_df.to_string(index=False))


if __name__ == "__main__":
    main()


def plot_descriptive_summary(df):
    """
    Create a single comprehensive figure showing all descriptive statistics
    """
    # Use only scored data
    scored = df[df["score"].notna()].copy()
    
    eastern_scores = scored[scored["category"] == "eastern"]["score"]
    western_scores = scored[scored["category"] == "western"]["score"]
    
    # Calculate statistics
    east_mean = eastern_scores.mean()
    west_mean = western_scores.mean()
    east_median = eastern_scores.median()
    west_median = western_scores.median()
    east_std = eastern_scores.std()
    west_std = western_scores.std()
    east_min = eastern_scores.min()
    east_max = eastern_scores.max()
    west_min = western_scores.min()
    west_max = western_scores.max()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Colors
    east_color = '#FF8C00'  # Orange
    west_color = '#1E90FF'  # Blue
    
    # Boxplots
    data = [eastern_scores, western_scores]
    positions = [1, 2.5]
    
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                    showmeans=True, meanline=True,
                    medianprops={'color': 'black', 'linewidth': 2},
                    meanprops={'color': 'red', 'linewidth': 2, 'linestyle': '--'})
    
    bp['boxes'][0].set_facecolor(east_color)
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor(west_color)
    bp['boxes'][1].set_alpha(0.6)
    
    # Add individual points
    x_east = np.random.normal(positions[0], 0.1, size=len(eastern_scores))
    x_west = np.random.normal(positions[1], 0.1, size=len(western_scores))
    
    ax.scatter(x_east, eastern_scores, alpha=0.3, s=30, color='black', edgecolor='white', linewidth=0.5)
    ax.scatter(x_west, western_scores, alpha=0.3, s=30, color='black', edgecolor='white', linewidth=0.5)
    
    # Highlight extremes
    ax.scatter([positions[0], positions[0], positions[1], positions[1]], 
               [east_min, east_max, west_min, west_max], 
               s=200, color=['red', 'green', 'red', 'green'], 
               edgecolor='black', linewidth=2, zorder=5)
    
    # Add mean markers
    ax.scatter([positions[0], positions[1]], [east_mean, west_mean], 
               s=200, color='blue', marker='d', edgecolor='black', linewidth=2, zorder=5)
    
    # Statistics boxes
    east_stats = f"EASTERN (n=62)\nMean: {east_mean:.3f}\nMedian: {east_median:.3f}\nSD: {east_std:.3f}\nMin: {east_min:.3f}\nMax: {east_max:.3f}\nRange: {east_max-east_min:.2f}"
    west_stats = f"WESTERN (n=57)\nMean: {west_mean:.3f}\nMedian: {west_median:.3f}\nSD: {west_std:.3f}\nMin: {west_min:.3f}\nMax: {west_max:.3f}\nRange: {west_max-west_min:.2f}"
    
    ax.text(positions[0]-0.3, 8.0, east_stats, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=east_color, alpha=0.3), ha='left', va='top')
    ax.text(positions[1]+0.3, 8.0, west_stats, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=west_color, alpha=0.3), ha='right', va='top')
    
    # Key insights
    diff = east_mean - west_mean
    insight_text = f"KEY INSIGHTS:\n• Mean diff: {abs(diff):.3f}\n• Eastern SD ({east_std:.2f}) > Western ({west_std:.2f})\n• Eastern range: {east_max-east_min:.2f}\n• Highest: Eastern ({east_max:.2f})\n• Lowest: Eastern ({east_min:.2f})"
    ax.text(0.5, -0.1, insight_text, transform=ax.transAxes, fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Labels
    ax.set_xticks(positions)
    ax.set_xticklabels(['Eastern', 'Western'], fontsize=14, fontweight='bold')
    ax.set_ylabel('Happiness Score (1-9)', fontsize=12)
    ax.set_title('Descriptive Statistics: Eastern vs Western Aesthetic Concepts', fontsize=16, fontweight='bold')
    ax.set_ylim(3.0, 9.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Save figure
    output_path = FIGURES_DIR / "descriptive_statistics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Figure saved to: {output_path}")
    
    return output_path


def main():
    df = load_scored_data()
    
  
    plot_descriptive_summary(df)
    
    print("\nDone.")


if __name__ == "__main__":
    main()