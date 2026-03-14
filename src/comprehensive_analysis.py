"""
Comprehensive Statistical Analysis: Eastern vs Western Aesthetic Concepts
Compares happiness scores between Eastern and Western aesthetic concepts in Met artwork titles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('default')
sns.set_palette("husl")

# Create directories if they don't exist
BASE_DIR = Path(__file__).parent.parent
FIGURES_DIR = BASE_DIR / "figures"
TABLES_DIR = BASE_DIR / "tables"
FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("COMPREHENSIVE STATISTICAL ANALYSIS: EASTERN VS WESTERN AESTHETIC CONCEPTS")
print("=" * 70)

def load_data():
    """Load the scored artwork data"""
    data_path = BASE_DIR / "data/processed/met_aesthetic_scored.csv"
    print(f"📂 Loading data...")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please run score_artworks.py first")
        exit(1)
    
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['score']).copy()  # Changed from happiness_score to score
    print(f"✅ Loaded {len(df)} scored artworks")
    
    # Check category distribution
    print(f"Categories: {dict(df['category'].value_counts())}")
    print()
    
    return df

def descriptive_statistics(df):
    """Calculate descriptive statistics for each category"""
    print("📊 DESCRIPTIVE STATISTICS")
    print("=" * 50)
    
    results = []
    for category in ['western', 'eastern']:
        cat_df = df[df['category'] == category]['score']  # Changed from happiness_score to score
        stats_dict = {
            'category': category.capitalize(),
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
        results.append(stats_dict)
        
        print(f"\n{category.upper()}:")
        print(f"  Count: {len(cat_df)}")
        print(f"  Mean: {cat_df.mean():.3f}")
        print(f"  Median: {cat_df.median():.3f}")
        print(f"  Std Dev: {cat_df.std():.3f}")
        print(f"  Range: [{cat_df.min():.3f}, {cat_df.max():.3f}]")
        print(f"  IQR: {stats_dict['iqr']:.3f}")
    
    # Save to CSV
    stats_df = pd.DataFrame(results)
    stats_path = TABLES_DIR / "descriptive_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\n✅ Statistics saved to: {stats_path}")
    
    return stats_df

def bootstrap_ci(df, n_bootstrap=10000, ci=95):
    """Calculate bootstrap confidence intervals"""
    print("\n🎯 BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 50)
    
    results = []
    np.random.seed(42)  # For reproducibility
    
    for category in ['western', 'eastern']:
        cat_data = df[df['category'] == category]['score'].values  # Changed from happiness_score to score
        
        # Bootstrap
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(cat_data, size=len(cat_data), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        # Calculate confidence interval
        lower = np.percentile(bootstrap_means, (100 - ci) / 2)
        upper = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)
        
        results.append({
            'category': category.capitalize(),
            'mean': np.mean(cat_data),
            'ci_lower': lower,
            'ci_upper': upper,
            'ci_width': upper - lower
        })
        
        print(f"\n{category.upper()}:")
        print(f"  Mean: {np.mean(cat_data):.3f}")
        print(f"  {ci}% CI: [{lower:.3f}, {upper:.3f}]")
        print(f"  CI width: {upper - lower:.3f}")
    
    # Save to CSV
    ci_df = pd.DataFrame(results)
    ci_path = TABLES_DIR / "bootstrap_ci.csv"
    ci_df.to_csv(ci_path, index=False)
    print(f"\n✅ Bootstrap results saved to: {ci_path}")
    
    return ci_df

def hypothesis_tests(df):
    """Run statistical hypothesis tests"""
    print("\n🔬 HYPOTHESIS TESTS")
    print("=" * 50)
    
    western = df[df['category'] == 'western']['score'].values  # Changed from happiness_score to score
    eastern = df[df['category'] == 'eastern']['score'].values  # Changed from happiness_score to score
    
    # T-test
    t_stat, t_p = stats.ttest_ind(western, eastern, equal_var=False)
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_p = stats.mannwhitneyu(western, eastern, alternative='two-sided')
    
    # Cohen's d effect size
    n1, n2 = len(western), len(eastern)
    var1, var2 = np.var(western, ddof=1), np.var(eastern, ddof=1)
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohen_d = (np.mean(western) - np.mean(eastern)) / pooled_se
    
    print(f"\nWestern vs Eastern comparison:")
    print(f"  t-test: t={t_stat:.3f}, p={t_p:.4f} {'(significant)' if t_p < 0.05 else '(not significant)'}")
    print(f"  Mann-Whitney U: U={u_stat:.3f}, p={u_p:.4f} {'(significant)' if u_p < 0.05 else '(not significant)'}")
    print(f"  Cohen's d: {cohen_d:.3f} (effect size)")
    
    # Interpret effect size
    if abs(cohen_d) < 0.2:
        effect_interpretation = "negligible effect"
    elif abs(cohen_d) < 0.5:
        effect_interpretation = "small effect"
    elif abs(cohen_d) < 0.8:
        effect_interpretation = "medium effect"
    else:
        effect_interpretation = "large effect"
    print(f"  Effect size interpretation: {effect_interpretation}")
    
    # Save results
    test_results = pd.DataFrame([{
        'test': 't-test',
        'statistic': t_stat,
        'p_value': t_p,
        'significant': t_p < 0.05
    }, {
        'test': 'Mann-Whitney U',
        'statistic': u_stat,
        'p_value': u_p,
        'significant': u_p < 0.05
    }, {
        'test': "Cohen's d",
        'statistic': cohen_d,
        'p_value': None,
        'significant': None
    }])
    
    test_path = TABLES_DIR / "hypothesis_tests.csv"
    test_results.to_csv(test_path, index=False)
    print(f"\n✅ Test results saved to: {test_path}")
    
    return test_results

def create_publication_figures(df, ci_df):
    """Create publication-quality figures"""
    print("\n🎨 Creating publication figures...")
    
    # Figure 1: Boxplot with confidence intervals
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Boxplot
    sns.boxplot(x='category', y='score', data=df, ax=ax1, palette=['#E67E22', '#3498DB'])
    ax1.set_xlabel('Category', fontsize=12)
    ax1.set_ylabel('Happiness Score', fontsize=12)
    ax1.set_title('Distribution of Happiness Scores by Category', fontsize=14, fontweight='bold')
    
    # Add individual points for smaller datasets
    sns.stripplot(x='category', y='score', data=df, ax=ax1, 
                  color='black', alpha=0.3, size=3, jitter=0.2)
    
    # Confidence interval plot
    categories = ['Western', 'Eastern']
    means = ci_df['mean'].values
    errors = [(ci_df['mean'] - ci_df['ci_lower']).values, 
              (ci_df['ci_upper'] - ci_df['mean']).values]
    
    ax2.errorbar(categories, means, yerr=errors, fmt='o', 
                 color='black', capsize=10, capthick=2, markersize=10)
    ax2.set_xlabel('Category', fontsize=12)
    ax2.set_ylabel('Mean Happiness Score', fontsize=12)
    ax2.set_title('Mean Scores with 95% Confidence Intervals', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (mean, lower, upper) in enumerate(zip(means, ci_df['ci_lower'], ci_df['ci_upper'])):
        ax2.text(i, upper + 0.02, f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "figure1_boxplot_with_ci.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure 1 saved: {fig1_path}")
    plt.close()
    
    # Figure 2: Distribution comparison (histogram)
    plt.figure(figsize=(10, 6))
    
    eastern_scores = df[df['category'] == 'eastern']['score']
    western_scores = df[df['category'] == 'western']['score']
    
    plt.hist(eastern_scores, alpha=0.7, label='Eastern', bins=15, color='#E67E22', edgecolor='black')
    plt.hist(western_scores, alpha=0.7, label='Western', bins=15, color='#3498DB', edgecolor='black')
    
    plt.xlabel('Happiness Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Happiness Scores: Eastern vs Western', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    fig2_path = FIGURES_DIR / "figure2_distribution.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure 2 saved: {fig2_path}")
    plt.close()
    
    # Figure 3: Confidence interval comparison
    plt.figure(figsize=(8, 6))
    
    y_pos = np.arange(len(categories))
    plt.barh(y_pos, means, xerr=errors, color=['#E67E22', '#3498DB'], 
             capsize=5, height=0.6, alpha=0.8)
    
    plt.yticks(y_pos, categories)
    plt.xlabel('Mean Happiness Score', fontsize=12)
    plt.title('Comparison of Mean Happiness Scores with 95% CI', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (mean, lower, upper) in enumerate(zip(means, ci_df['ci_lower'], ci_df['ci_upper'])):
        plt.text(mean + 0.02, i, f'{mean:.3f}', va='center', fontsize=10)
    
    fig3_path = FIGURES_DIR / "figure3_ci_comparison.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure 3 saved: {fig3_path}")
    plt.close()

def main():
    """Main analysis function"""
    
    # Load data
    df = load_data()
    
    # Descriptive statistics
    stats_df = descriptive_statistics(df)
    
    # Bootstrap confidence intervals
    ci_df = bootstrap_ci(df)
    
    # Hypothesis tests
    test_results = hypothesis_tests(df)
    
    # Create figures
    create_publication_figures(df, ci_df)
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nFiles created:")
    print(f"  📊 Tables: {TABLES_DIR}")
    print(f"    - descriptive_stats.csv")
    print(f"    - bootstrap_ci.csv")
    print(f"    - hypothesis_tests.csv")
    print(f"\n  🖼️  Figures: {FIGURES_DIR}")
    print(f"    - figure1_boxplot_with_ci.png")
    print(f"    - figure2_distribution.png")
    print(f"    - figure3_ci_comparison.png")
    print("\nNext steps: Add these to your README.md!")

if __name__ == "__main__":
    main()
