"""
Task: Fetch and process artwork data from The Metropolitan Museum of Art API
This script downloads artwork data, saves raw JSON, and creates a processed CSV
for hedonometer analysis of artwork titles and descriptions.
"""

import requests
import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
import os

print("=" * 60)
print("MET MUSEUM API DATA ACQUISITION")
print("=" * 60)

# -----------------------------------------------------------------------------
# Step 1: Set up directories
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "met"
PROCESSED_DIR = ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print(f"Raw data directory: {RAW_DIR}")
print(f"Processed data directory: {PROCESSED_DIR}")

# -----------------------------------------------------------------------------
# Step 2: API configuration
# -----------------------------------------------------------------------------
BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"

# =============================================================================
# IMPROVED: Multiple search terms for better coverage
# =============================================================================
# Rationale: 
# 1. Increase sample size for statistical power
# 2. Include emotional words likely to match labMT
# 3. Enable comparisons across themes/themes
# 4. Capture diverse artistic traditions

search_terms = [
    "love",      # High labMT score (8.42)
    "death",     # Low labMT score (1.54)
    "war",       # Low labMT score (2.50)
    "peace",     # High labMT score (7.20)
    "nature",    # Medium labMT score (6.50)
    "beauty",    # High labMT score (7.80)
    "sorrow",    # Low labMT score (2.30)
    "joy",       # High labMT score (8.16)
    "flowers",   # Original term
    "landscape", # Art term with nature vocabulary
    "portrait",  # Art term with human subjects
    "religious"  # Art term with moral/spiritual vocabulary
]

objects_per_term = 30  # Total: 12 * 30 = 360 objects
max_total = 300  # Safety limit

print(f"\n Search terms: {search_terms}")
print(f" Objects per term: {objects_per_term}")
print(f" Target total: {min(len(search_terms) * objects_per_term, max_total)} objects")

# -----------------------------------------------------------------------------
# Step 3: Search and collect object IDs
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 1: Searching for objects")
print("=" * 60)

all_object_ids = []
search_results = {}

for term in search_terms:
    print(f"\n Searching for: '{term}'")
    
    search_url = f"{BASE_URL}/search"
    params = {
        "q": term,
        "hasImages": True
    }
    
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        object_ids = data.get("objectIDs", [])
        print(f"   Found {len(object_ids)} objects")
        
        # Take first N objects for this term
        term_ids = object_ids[:objects_per_term]
        all_object_ids.extend(term_ids)
        search_results[term] = len(term_ids)
        
        # Be nice to API
        time.sleep(0.5)
        
    except requests.exceptions.RequestException as e:
        print(f"   Error: {e}")
        search_results[term] = 0

# Remove duplicates while preserving order
seen = set()
unique_object_ids = []
for obj_id in all_object_ids:
    if obj_id not in seen and obj_id is not None:
        seen.add(obj_id)
        unique_object_ids.append(obj_id)

# Limit total
unique_object_ids = unique_object_ids[:max_total]

print(f"\n Total unique objects found: {len(unique_object_ids)}")
print(f" Search results by term:")
for term, count in search_results.items():
    print(f"  - {term}: {count} objects")

# -----------------------------------------------------------------------------
# Step 4: Download object details
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 2: Downloading object details")
print("=" * 60)

artworks = []
failed_ids = []

for i, obj_id in enumerate(unique_object_ids):
    print(f"  Downloading {i+1}/{len(unique_object_ids)}: Object {obj_id}", end="")
    
    detail_url = f"{BASE_URL}/objects/{obj_id}"
    
    try:
        detail_response = requests.get(detail_url)
        detail_response.raise_for_status()
        
        artwork = detail_response.json()
        artworks.append(artwork)
        print(" ✅")
        
    except requests.exceptions.RequestException as e:
        print(f"  Failed: {e}")
        failed_ids.append(obj_id)
    
    time.sleep(0.2)

print(f"\n Successfully downloaded: {len(artworks)} objects")
print(f" Failed downloads: {len(failed_ids)} objects")

# -----------------------------------------------------------------------------
# Step 5: Save raw data with provenance
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 3: Saving raw data")
print("=" * 60)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
raw_filename = f"met_objects_multi_{timestamp}.json"
raw_path = RAW_DIR / raw_filename

with open(raw_path, "w", encoding="utf-8") as f:
    json.dump(artworks, f, indent=2, ensure_ascii=False)

print(f"Raw data saved to: {raw_path}")
print(f"File size: {raw_path.stat().st_size / 1024:.1f} KB")

# -----------------------------------------------------------------------------
# Step 6: Create processed DataFrame
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 4: Creating processed dataset")
print("=" * 60)

if not artworks:
    print("No artworks downloaded. Exiting.")
    exit()

df = pd.DataFrame(artworks)
print(f"Total columns in raw data: {len(df.columns)}")

# Select columns relevant for text analysis
text_columns = ['title', 'artistDisplayName', 'objectDate', 'medium', 'classification', 
                'department', 'culture', 'period', 'dynasty', 'reign', 'objectName']

keep_cols = [col for col in text_columns if col in df.columns]
df_processed = df[keep_cols].copy()

# Add metadata
df_processed['object_id'] = df.get('objectID', range(len(df)))
df_processed['is_public_domain'] = df.get('isPublicDomain', False)
df_processed['accession_year'] = df.get('accessionYear', None)

# Create analysis text
df_processed['analysis_text'] = df_processed['title'].fillna('') + ' ' + df_processed['medium'].fillna('')
df_processed['text_length'] = df_processed['analysis_text'].str.len()

# Add search term tag (for grouping analysis)
def tag_search_term(title):
    title_lower = str(title).lower()
    for term in search_terms:
        if term in title_lower:
            return term
    return 'other'

df_processed['search_term'] = df_processed['title'].apply(tag_search_term)

print(f"\nProcessed dataset shape: {df_processed.shape}")
print(f"\nArtworks by search term:")
print(df_processed['search_term'].value_counts())

# -----------------------------------------------------------------------------
# Step 7: Save processed data
# -----------------------------------------------------------------------------
processed_filename = f"met_artworks_multi_{timestamp}.csv"
processed_path = PROCESSED_DIR / processed_filename

df_processed.to_csv(processed_path, index=False)
print(f"\n💾 Processed data saved to: {processed_path}")

# -----------------------------------------------------------------------------
# Step 8: Create data dictionary
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 5: Creating data dictionary")
print("=" * 60)

data_dict = pd.DataFrame({
    'column': df_processed.columns,
    'dtype': df_processed.dtypes.astype(str),
    'non_null_count': df_processed.count().values,
    'example': [df_processed[col].iloc[0] if len(df_processed) > 0 else None for col in df_processed.columns]
})

descriptions = {
    'title': 'Title of the artwork',
    'artistDisplayName': 'Name of the artist',
    'objectDate': 'Date or period when object was created',
    'medium': 'Materials used in the artwork',
    'classification': 'Art classification',
    'department': 'Museum department',
    'culture': 'Cultural origin',
    'period': 'Historical period',
    'dynasty': 'Dynasty (for Chinese/Egyptian art)',
    'reign': 'Reign of ruler',
    'objectName': 'Type of object',
    'object_id': 'Unique identifier from MET',
    'is_public_domain': 'Whether artwork is in public domain',
    'accession_year': 'Year museum acquired the object',
    'analysis_text': 'Combined text for hedonometer analysis',
    'text_length': 'Length of analysis text in characters',
    'search_term': 'Term used to discover this artwork'
}

data_dict['description'] = data_dict['column'].map(descriptions).fillna('No description')
dict_path = PROCESSED_DIR / "met_data_dictionary.csv"
data_dict.to_csv(dict_path, index=False)
print(f"Data dictionary saved to: {dict_path}")

# -----------------------------------------------------------------------------
# Step 9: Provenance and ethics statement
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 6: Data provenance")
print("=" * 60)

provenance = f"""
MET MUSEUM DATA ACQUISITION SUMMARY
====================================
Download Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Script: fetch_met_data.py
API: Metropolitan Museum of Art Collection API

SEARCH METHODOLOGY
------------------
Rationale: Multiple search terms were used to:
1. Increase sample size for statistical power
2. Include emotional words likely to match labMT (love, death, joy, sorrow)
3. Enable comparisons across themes (nature vs religious vs war)
4. Capture diverse artistic traditions

Search terms: {search_terms}
Objects per term: {objects_per_term}
Total unique objects requested: {len(unique_object_ids)}
Successfully downloaded: {len(artworks)}
Failed downloads: {len(failed_ids)}

SAMPLE CHARACTERISTICS
----------------------
Artworks by search term:
{df_processed['search_term'].value_counts().to_string()}

Date range: {df_processed['objectDate'].min()} to {df_processed['objectDate'].max()}
Unique artists: {df_processed['artistDisplayName'].nunique()}
Unique departments: {df_processed['department'].nunique() if 'department' in df_processed.columns else 'N/A'}

LIMITATIONS
-----------
- Only includes objects with images
- Limited to first {objects_per_term} results per term
- Search terms may bias toward certain artistic traditions
- Text may contain historical language and terminology
- Not representative of all museum collections
"""

print(provenance)

provenance_path = PROCESSED_DIR / "met_data_provenance.txt"
with open(provenance_path, "w") as f:
    f.write(provenance)
print(f"Provenance saved to: {provenance_path}")

# -----------------------------------------------------------------------------
# Step 10: Preview
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("DATA PREVIEW")
print("=" * 60)

print("\nFirst 5 artworks:")
print(df_processed[['title', 'artistDisplayName', 'search_term', 'text_length']].head())

print(f"\nText length statistics:")
print(df_processed['text_length'].describe())

print("\n" + "=" * 60)
print("MET DATA ACQUISITION COMPLETE!")
print("=" * 60)
print("\nNext steps:")
print("1. Run hedonometer scoring on 'analysis_text' column")
print("2. Compare scores across search terms, departments, or periods")
print("3. Create visualizations of your findings")



