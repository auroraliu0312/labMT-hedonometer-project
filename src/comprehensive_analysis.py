"""
Comprehensive Statistical Analysis: Eastern vs Western Aesthetic Concepts
Includes: descriptive stats, confidence intervals, bootstrap, hypothesis testing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
base_dir = Path(__file__).parent.parent
data_path = base_dir / "data/processed/met_aesthetic_scored.csv"
figures_dir = base_dir / "figures"
tables_dir = base_dir / "tables"

# Create directories
figures_dir.mkdir(exist_ok=True)
tables_dir.mkdir(exist_ok=True)

def load_data():
    """Load the scored artwork data"""
    print("📂 Loading data...")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['happiness_score']).copy()
    print(f"✅ Loaded {len(df)} scored artworks")
    return df

def descriptive_statistics(df):
    """Calculate and save descriptive statistics"""
    print("\n📊 DESCRIPTIVE STATISTICS")
    print("=" * 50)
    
    stats_list = []
    for category in ['western', 'eastern']:
        cat_df = df[df['category'] == category]['happiness_score']
        
        # Calculate statistics
        stats_dict = {
            'category': category,
            'count': len(cat_df),
            'mean': cat_df.mean(),
            'median': cat_df.median(),
            'std': cat_df.std(),
            'min': cat_df.min(),
            'max': cat_df.max(),
            'q1': cat_df.quantile(0.25),
            'q3': cat_df.quantile(0.75),
            'iqr': cat_df.quantile(0.75) - cat_df.quantile(0.25)
        }
        stats_list.append(stats_dict)
        
        print(f"\n{category.upper()}:")
        print(f"  Count: {stats_dict['count']}")
        print(f"  Mean: {stats_dict['mean']:.3f}")
        print(f"  Median: {stats_dict['median']:.3f}")
        print(f"  Std Dev: {stats_dict['std']:.3f}")
        print(f"  Range: [{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]")
        print(f"  IQR: {stats_dict['iqr']:.3f}")
    
    # Save to CSV
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(tables_dir / "descriptive_stats.csv", index=False)
    print(f"\n✅ Statistics saved to: {tables_dir}/descriptive_stats.csv")
    
    return stats_df

def confidence_intervals(df, n_bootstrap=10000, confidence=0.95):
    """Calculate bootstrap confidence intervals"""
    print("\n🎯 BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 50)
    
    results = []
    
    for category in ['western', 'eastern']:
        cat_data = df[df['category'] == category]['happiness_score'].values
        
        # Bootstrap
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(cat_data, size=len(cat_data), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        # Calculate confidence interval
        lower = np.percentile(bootstrap_means, (1-confidence)/2 * 100)
        upper = np.percentile(bootstrap_means, (1 + confidence)/2 * 100)
        
        results.append({
            'category': category,
            'mean': np.mean(cat_data),
            'ci_lower': lower,
            'ci_upper': upper,
            'ci_width': upper - lower
        })
        
        print(f"\n{category.upper()}:")
        print(f"  Mean: {np.mean(cat_data):.3f}")
        print(f"  {confidence*100:.0f}% CI: [{lower:.3f}, {upper:.3f}]")
        print(f"  CI width: {upper-lower:.3f}")
    
    # Save to CSV
    ci_df = pd.DataFrame(results)
    ci_df.to_csv(tables_dir / "confidence_intervals.csv", index=False)
    
    return ci_df

def hypothesis_tests(df):
    """Run statistical tests to compare groups"""
    print("\n🔬 HYPOTHESIS TESTS")
    print("=" * 50)
    
    western = df[df['category'] == 'western']['happiness_score']
    eastern = df[df['category'] == 'eastern']['happiness_score']
    
    # 1. Independent t-test (parametric)
    t_stat, t_pval = stats.ttest_ind(western, eastern, equal_var=False)
    
    # 2. Mann-Whitney U test (non-parametric)
    u_stat, u_pval = stats.mannwhitneyu(western, eastern, alternative='two-sided')
    
    # 3. Cohen's d effect size
    pooled_std = np.sqrt((western.std()**2 + eastern.std()**2) / 2)
    cohens_d = (western.mean() - eastern.mean()) / pooled_std
    
    results = {
        'test': ['t-test', 'Mann-Whitney U', "Cohen's d"],
        'statistic': [t_stat, u_stat, cohens_d],
        'p_value': [t_pval, u_pval, np.nan],
        'significant_at_0.05': [t_pval < 0.05, u_pval < 0.05, np.nan]
    }
    
    results_df = pd.DataFrame(results)
    
    print("\nWestern vs Eastern comparison:")
    print(f"  t-test: t={t_stat:.3f}, p={t_pval:.4f} {'(SIGNIFICANT)' if t_pval < 0.05 else '(not significant)'}")
    print(f"  Mann-Whitney U: U={u_stat:.3f}, p={u_pval:.4f} {'(SIGNIFICANT)' if u_pval < 0.05 else '(not significant)'}")
    print(f"  Cohen's d: {cohens_d:.3f} (effect size)")
    
    # Interpretation
    if abs(cohens_d) < 0.2:
        effect = "negligible"
    elif abs(cohens_d) < 0.5:
        effect = "small"
    elif abs(cohens_d) < 0.8:
        effect = "medium"
    else:
        effect = "large"
    
    print(f"  Effect size interpretation: {effect} effect")
    
    results_df.to_csv(tables_dir / "hypothesis_tests.csv", index=False)
    return results_df

def create_publication_figures(df, ci_df):
    """Create all figures for the README"""
    
    # Figure 1: Enhanced boxplot with confidence intervals
    plt.figure(figsize=(12, 7))
    
    # Boxplot
    ax = sns.boxplot(x='category', y='happiness_score', data=df, 
                     palette=['#3498db', '#2ecc71'], width=0.6)
    
    # Add individual points
    sns.stripplot(x='category', y='happiness_score', data=df, 
                  color='black', alpha=0.4, size=3, jitter=0.2)
    
    # Add confidence intervals
    for i, category in enumerate(['western', 'eastern']):
        ci = ci_df[ci_df['category'] == category].iloc[0]
        plt.plot([i-0.2, i+0.2], [ci['mean'], ci['mean']], 
                color='red', linewidth=2, linestyle='-')
        plt.plot([i, i], [ci['ci_lower'], ci['ci_upper']], 
                color='red', linewidth=2, linestyle='-')
        plt.plot(i-0.1, ci['ci_lower'], 'v', color='red', markersize=8)
        plt.plot(i-0.1, ci['ci_upper'], '^', color='red', markersize=8)
    
    # Add reference line at neutral (5.0)
    plt.axhline(y=5.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Labels and title
    plt.title('Happiness Scores in Met Museum Artwork Titles:\nEastern vs. Western Aesthetic Concepts', 
              fontsize=16, pad=20)
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Happiness Score (labMT 1-9 scale)', fontsize=14)
    
    # Custom x-tick labels with counts
    counts = df['category'].value_counts()
    ax.set_xticklabels([f'Western\n(n={counts["western"]})', 
                        f'Eastern\n(n={counts["eastern"]})'])
    
    # Add legend explanation
    plt.text(0.02, 0.98, 'Red line: Mean with 95% CI', 
             transform=ax.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(figures_dir / "figure1_boxplot_with_ci.png", dpi=300, bbox_inches='tight')
    print(f"✅ Figure 1 saved: {figures_dir}/figure1_boxplot_with_ci.png")
    
    # Figure 2: Distribution histogram
    plt.figure(figsize=(12, 6))
    
    plt.hist(df[df['category']=='western']['happiness_score'], 
             alpha=0.7, label='Western', bins=12, color='#3498db', 
             edgecolor='black', density=True)
    plt.hist(df[df['category']=='eastern']['happiness_score'], 
             alpha=0.7, label='Eastern', bins=12, color='#2ecc71', 
             edgecolor='black', density=True)
    
    plt.title('Distribution of Happiness Scores:\nEastern vs. Western Aesthetic Concepts', 
              fontsize=16, pad=20)
    plt.xlabel('Happiness Score', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=12)
    
    # Add vertical lines for means
    plt.axvline(df[df['category']=='western']['happiness_score'].mean(), 
                color='#3498db', linestyle='--', linewidth=2, alpha=0.8)
    plt.axvline(df[df['category']=='eastern']['happiness_score'].mean(), 
                color='#2ecc71', linestyle='--', linewidth=2, alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "figure2_distribution.png", dpi=300, bbox_inches='tight')
    print(f"✅ Figure 2 saved: {figures_dir}/figure2_distribution.png")
    
    # Figure 3: Confidence interval comparison
    plt.figure(figsize=(10, 6))
    
    y_pos = [1, 0]
    colors = ['#3498db', '#2ecc71']
    
    for i, (_, row) in enumerate(ci_df.iterrows()):
        plt.errorbar(row['mean'], i, xerr=[[row['mean']-row['ci_lower']], [row['ci_upper']-row['mean']]], 
                    fmt='o', color=colors[i], capsize=5, capthick=2, markersize=10)
        plt.text(row['mean']+0.1, i, f'{row["mean"]:.2f}', va='center', fontsize=11)
    
    plt.yticks(y_pos, ['Western', 'Eastern'])
    plt.xlabel('Happiness Score', fontsize=14)
    plt.title('Comparison of Means with 95% Confidence Intervals', fontsize=16, pad=20)
    plt.axvline(x=5.0, color='gray', linestyle='--', alpha=0.5)
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "figure3_ci_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✅ Figure 3 saved: {figures_dir}/figure3_ci_comparison.png")

def create_example_table(df):
    """Create a table of interesting examples for the README"""
    
    examples = []
    
    # Highest scoring Western
    top_western = df[df['category']=='western'].nlargest(3, 'happiness_score')
    for _, row in top_western.iterrows():
        examples.append({
            'title': row['title'],
            'category': 'Western',
            'score': f"{row['happiness_score']:.2f}",
            'note': 'Highest in category'
        })
    
    # Highest scoring Eastern
    top_eastern = df[df['category']=='eastern'].nlargest(3, 'happiness_score')
    for _, row in top_eastern.iterrows():
        examples.append({
            'title': row['title'],
            'category': 'Eastern',
            'score': f"{row['happiness_score']:.2f}",
            'note': 'Highest in category'
        })
    
    # Lowest scoring Western
    bottom_western = df[df['category']=='western'].nsmallest(3, 'happiness_score')
    for _, row in bottom_western.iterrows():
        examples.append({
            'title': row['title'],
            'category': 'Western',
            'score': f"{row['happiness_score']:.2f}",
            'note': 'Lowest in category'
        })
    
    # Lowest scoring Eastern
    bottom_eastern = df[df['category']=='eastern'].nsmallest(3, 'happiness_score')
    for _, row in bottom_eastern.iterrows():
        examples.append({
            'title': row['title'],
            'category': 'Eastern',
            'score': f"{row['happiness_score']:.2f}",
            'note': 'Lowest in category'
        })
    
    examples_df = pd.DataFrame(examples)
    examples_df.to_csv(tables_dir / "notable_examples.csv", index=False)
    return examples_df

def main():
    print("=" * 70)
    print("COMPREHENSIVE STATISTICAL ANALYSIS: EASTERN VS WESTERN AESTHETIC CONCEPTS")
    print("=" * 70)
    
    # Load data
    df = load_data()
    
    # Run analyses
    stats_df = descriptive_statistics(df)
    ci_df = confidence_intervals(df, n_bootstrap=10000)
    tests_df = hypothesis_tests(df)
    
    # Create figures
    print("\n🎨 Creating publication figures...")
    create_publication_figures(df, ci_df)
    
    # Create example table
    examples_df = create_example_table(df)
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nFiles created:")
    print(f"  📊 Tables: {tables_dir}")
    print(f"  🖼️  Figures: {figures_dir}")
    print("\nNext steps: Add these to your README.md!")

if __name__ == "__main__":
    main()

