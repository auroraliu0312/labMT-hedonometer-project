"""
MET Museum API Data Acquisition
Fetches artwork data based on emotional keywords for hedonometer analysis
"""

import requests
import time
import pandas as pd
import os
from pathlib import Path
import json
from datetime import datetime

# Create directories
RAW_DIR = Path("data/raw/met")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("MET MUSEUM API DATA ACQUISITION")
print("=" * 70)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Raw data directory: {RAW_DIR.absolute()}")
print(f"Processed data directory: {PROCESSED_DIR.absolute()}")
print()

# Search terms related to emotional concepts
search_terms = [
    'love', 'death', 'war', 'peace', 'nature', 
    'beauty', 'sorrow', 'joy', 'flowers', 'landscape',
    'portrait', 'religious', 'happiness', 'sadness',
    'victory', 'defeat', 'celebration', 'mourning'
]

# Limit objects per term to be kind to the API
objects_per_term = 15
print(f"Search terms: {len(search_terms)} terms")
print(f"Search terms list: {search_terms}")
print(f"Objects per term: {objects_per_term}")
print(f"Target total: {len(search_terms) * objects_per_term} objects")
print()

# Store all object IDs we find
all_object_ids = set()
search_log = []

print("=" * 70)
print("STEP 1: SEARCHING FOR OBJECTS BY KEYWORD")
print("=" * 70)

for term in search_terms:
    print(f"\n🔍 Searching for: '{term}'")
    
    # API search endpoint - add hasImages=true to get only objects with images
    search_url = f"https://collectionapi.metmuseum.org/public/collection/v1/search?q={term}&hasImages=true"
    
    try:
        response = requests.get(search_url)
        if response.status_code == 200:
            data = response.json()
            object_ids = data.get('objectIDs', [])
            
            # Log the search
            search_log.append({
                'term': term,
                'timestamp': datetime.now().isoformat(),
                'total_found': len(object_ids) if object_ids else 0,
                'selected': min(objects_per_term, len(object_ids)) if object_ids else 0
            })
            
            if object_ids and len(object_ids) > 0:
                # Take only the first N objects
                selected = object_ids[:min(objects_per_term, len(object_ids))]
                all_object_ids.update(selected)
                print(f"  ✅ Found {len(object_ids)} objects, selected {len(selected)}")
                print(f"  📊 Sample IDs: {selected[:3]}")
            else:
                print(f"  ⚠️  No objects found")
        else:
            print(f"  ❌ API error: {response.status_code}")
            search_log.append({
                'term': term,
                'timestamp': datetime.now().isoformat(),
                'error': f"HTTP {response.status_code}"
            })
    
    except Exception as e:
        print(f"  ❌ Error: {e}")
        search_log.append({
            'term': term,
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        })
    
    # Be nice to the API - wait between requests
    time.sleep(0.3)

print(f"\n✅ Total unique objects collected: {len(all_object_ids)}")

# Save search log
search_log_df = pd.DataFrame(search_log)
search_log_file = RAW_DIR / "met_search_log.csv"
search_log_df.to_csv(search_log_file, index=False)
print(f"✅ Saved search log to: {search_log_file}")

# Save the object IDs
object_ids_file = RAW_DIR / "met_object_ids.csv"
pd.DataFrame({"object_id": list(all_object_ids)}).to_csv(object_ids_file, index=False)
print(f"✅ Saved object IDs to: {object_ids_file}")
print()

print("=" * 70)
print("STEP 2: FETCHING OBJECT DETAILS")
print("=" * 70)
print(f"Estimated time: {len(all_object_ids) * 0.25:.1f} seconds...")
print()

# Fetch details for each object
artworks = []
object_ids_list = list(all_object_ids)
fetch_log = []

for i, obj_id in enumerate(object_ids_list):
    if i % 10 == 0:
        print(f"  Progress: {i}/{len(object_ids_list)} objects fetched")
    
    detail_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}"
    
    try:
        response = requests.get(detail_url)
        if response.status_code == 200:
            data = response.json()
            
            # Extract relevant fields
            artwork = {
                'object_id': data.get('objectID'),
                'title': data.get('title'),
                'department': data.get('department'),
                'classification': data.get('classification'),
                'culture': data.get('culture'),
                'period': data.get('period'),
                'object_date': data.get('objectDate'),
                'object_begin': data.get('objectBeginDate'),
                'object_end': data.get('objectEndDate'),
                'medium': data.get('medium'),
                'artist_name': data.get('artistDisplayName'),
                'artist_nationality': data.get('artistNationality'),
                'artist_begin': data.get('artistBeginDate'),
                'artist_end': data.get('artistEndDate'),
                'accession_year': data.get('accessionYear'),
                'tags': json.dumps([tag['term'] for tag in data.get('tags', [])]) if data.get('tags') else None
            }
            artworks.append(artwork)
            
            fetch_log.append({
                'object_id': obj_id,
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'has_title': bool(data.get('title'))
            })
        else:
            fetch_log.append({
                'object_id': obj_id,
                'timestamp': datetime.now().isoformat(),
                'status': f'error_{response.status_code}',
                'has_title': False
            })
            print(f"  ⚠️  Error fetching {obj_id}: HTTP {response.status_code}")
    except Exception as e:
        fetch_log.append({
            'object_id': obj_id,
            'timestamp': datetime.now().isoformat(),
            'status': f'error_{str(e)[:50]}',
            'has_title': False
        })
        print(f"  ❌ Error fetching {obj_id}: {e}")
    
    # Rate limiting
    time.sleep(0.2)

print(f"\n✅ Successfully fetched {len(artworks)} artworks")

# Save fetch log
fetch_log_df = pd.DataFrame(fetch_log)
fetch_log_file = RAW_DIR / "met_fetch_log.csv"
fetch_log_df.to_csv(fetch_log_file, index=False)
print(f"✅ Saved fetch log to: {fetch_log_file}")

# Create DataFrame
df = pd.DataFrame(artworks)

# Save raw data
raw_file = RAW_DIR / "met_artworks_raw.csv"
df.to_csv(raw_file, index=False)
print(f"✅ Saved raw data to: {raw_file}")

print("\n" + "=" * 70)
print("STEP 3: PROCESSING DATA")
print("=" * 70)

# Basic cleaning for processed data
df_processed = df.copy()
initial_rows = len(df_processed)

# Remove rows with no title
df_processed = df_processed[df_processed['title'].notna()]
print(f"✅ Removed {initial_rows - len(df_processed)} rows with missing titles")

# Create century column from object_begin
def get_century(year):
    if pd.isna(year) or year <= 0:
        return None
    return (int(year) // 100) * 100

df_processed['century'] = df_processed['object_begin'].apply(get_century)
century_count = df_processed['century'].notna().sum()
print(f"✅ Created century field for {century_count} artworks")

# Simple title cleaning (remove punctuation)
df_processed['title_clean'] = df_processed['title'].str.replace(r'[^\w\s]', '', regex=True)
print(f"✅ Created cleaned titles for hedonometer scoring")

# Count missing values for documentation
missing_counts = df_processed.isna().sum().to_dict()

# Save processed data
processed_file = PROCESSED_DIR / "met_artworks_processed.csv"
df_processed.to_csv(processed_file, index=False)
print(f"✅ Saved processed data to: {processed_file}")

print("\n" + "=" * 70)
print("DATA SUMMARY")
print("=" * 70)
print(f"Total artworks: {len(df_processed)}")
print(f"Total columns: {len(df_processed.columns)}")
print(f"Columns: {', '.join(df_processed.columns)}")
print(f"\n📊 Department distribution:")
dept_stats = df_processed['department'].value_counts().head(5)
for dept, count in dept_stats.items():
    print(f"  • {dept}: {count} artworks")

print(f"\n📊 Century distribution:")
century_stats = df_processed['century'].value_counts().head(5)
for century, count in century_stats.items():
    print(f"  • {century}s: {count} artworks")

print(f"\n📊 Culture distribution:")
culture_stats = df_processed['culture'].value_counts().head(5)
for culture, count in culture_stats.items():
    print(f"  • {culture}: {count} artworks")

print(f"\n📊 Missing data summary:")
for col in ['culture', 'artist_name', 'artist_nationality', 'period']:
    missing = df_processed[col].isna().sum()
    pct = (missing / len(df_processed)) * 100
    print(f"  • {col}: {missing} missing ({pct:.1f}%)")

print("\n" + "=" * 70)
print("✅ DATA ACQUISITION COMPLETE")
print("=" * 70)
print("\nNext steps: Run scoring with labMT word list")
print("> python src/02_score_titles.py")



