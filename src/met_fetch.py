"""
Met Museum: Eastern vs. Western Aesthetic Concepts
Research Question: How do happiness scores differ between Eastern and Western 
aesthetic concepts found in Met artwork titles?
"""

import requests
import pandas as pd
import time
import re
from pathlib import Path

# Create folders
base_dir = Path(__file__).parent.parent  # Goes up one level from src/
(base_dir / "data/raw").mkdir(parents=True, exist_ok=True)
(base_dir / "data/processed").mkdir(parents=True, exist_ok=True)

# ============================================
# EASTERN VS WESTERN AESTHETIC CONCEPTS
# ============================================

SEARCH_TERMS = {
    "western": [
        "beauty", "sublime", "pastoral", "romantic", "ideal", 
        "grace", "glory", "divine", "harmony", "splendor"
    ],
    "eastern": [
        "zen", "ukiyo", "wabi sabi", "mono no aware", "feng shui",
        "simplicity", "impermanence", "emptiness", "enlightenment", 
        "meditation", "bamboo", "cherry blossom", "lotus", "nirvana"
    ]
}

def search_met(term, max_results=15):
    """Search Met API for a term"""
    url = "https://collectionapi.metmuseum.org/public/collection/v1/search"
    params = {"q": term, "hasImages": "true"}
    
    try:
        print(f"    Searching for '{term}'...")
        response = requests.get(url, params=params)
        data = response.json()
        object_ids = data.get('objectIDs', [])
        if object_ids:
            print(f"      Found {len(object_ids[:max_results])} objects")
            return object_ids[:max_results]
        return []
    except Exception as e:
        print(f"      Error: {e}")
        return []

def get_object(obj_id):
    """Get object details"""
    url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}"
    try:
        response = requests.get(url)
        return response.json()
    except:
        return None

def main():
    """Main function to collect artworks"""
    print("=" * 60)
    print("MET MUSEUM: EASTERN VS WESTERN AESTHETIC CONCEPTS")
    print("=" * 60)
    
    all_artworks = []
    total_found = 0
    
    for category, terms in SEARCH_TERMS.items():
        print(f"\n📁 {category.upper()} concepts ({len(terms)} terms)")
        
        for term in terms:
            object_ids = search_met(term)
            
            for obj_id in object_ids:
                data = get_object(obj_id)
                if data and data.get('title'):
                    all_artworks.append({
                        'object_id': obj_id,
                        'title': data['title'],
                        'category': category,
                        'term_used': term,
                        'department': data.get('department', ''),
                        'culture': data.get('culture', ''),
                        'period': data.get('period', ''),
                        'artist': data.get('artistDisplayName', ''),
                        'date': data.get('objectDate', ''),
                        'object_begin': data.get('objectBeginDate', '')
                    })
                    total_found += 1
                    if total_found % 10 == 0:
                        print(f"      Collected {total_found} artworks so far...")
                
                # Small delay to be nice to API
                time.sleep(0.1)
            
            # Delay between terms
            time.sleep(0.2)
    
    # Remove duplicates
    df = pd.DataFrame(all_artworks)
    df = df.drop_duplicates(subset=['object_id'])
    
    # Save raw data
    output_path = base_dir / "data/raw/met_aesthetic_raw.csv"
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print(f"✅ COLLECTION COMPLETE!")
    print("=" * 60)
    print(f"Total unique artworks: {len(df)}")
    print(f"  - Western concepts: {len(df[df['category']=='western'])}")
    print(f"  - Eastern concepts: {len(df[df['category']=='eastern'])}")
    print(f"\nData saved to: {output_path}")
    
    # Show sample
    print("\n📋 First 5 artworks collected:")
    print(df[['title', 'category', 'term_used', 'culture']].head())
    
    # Quick summary
    print("\n📊 Summary by category:")
    summary = df.groupby('category').size().reset_index(name='count')
    print(summary)

if __name__ == "__main__":
    main()


