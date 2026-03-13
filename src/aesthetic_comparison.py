import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. PATH TO YOUR DATA
file_path = 'data/processed/met_aesthetic_scored.csv'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print("Data loaded! Found the 'score' column.")
else:
    print(f"Error: Cannot find {file_path}")
    exit()

# 2. LABELLING (East vs West)
east_keywords = ['zen', 'lotus', 'buddha', 'china', 'japan', 'ink', 'dynasty', 'silk', 'calligraphy']
west_keywords = ['oil', 'canvas', 'portrait', 'renaissance', 'baroque', 'gothic', 'europe', 'france']

def label_aesthetic(text):
    text = str(text).lower()
    if any(word in text for word in east_keywords):
        return 'Eastern Aesthetic'
    elif any(word in text for word in west_keywords):
        return 'Western Aesthetic'
    return 'Other'

df['Aesthetic'] = df['title'].apply(label_aesthetic)
comparison_df = df[df['Aesthetic'] != 'Other']

# 3. THE PLOT
plt.figure(figsize=(10, 6))
# Using 'score' because that's what terminal showed!
sns.boxplot(x='Aesthetic', y='score', data=comparison_df, palette="Set2", showfliers=False)
sns.stripplot(x='Aesthetic', y='score', data=comparison_df, color="black", alpha=0.3)

plt.title('Happiness Scores: Eastern vs. Western Aesthetics in the Met', fontsize=14)
plt.ylabel('labMT Happiness Score')
plt.xlabel('Aesthetic Category')

# 4. SAVE
plt.savefig('figures/east_west_comparison.png')
print("Success! Chart saved to: figures/east_west_comparison.png")

