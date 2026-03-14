
### Setup Steps
```bash
# 1. Clone repository
git clone https://github.com/auroraliu0312/labMT-hedonometer-project.git
cd labMT-hedonometer-project

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Mac/Linux
# .venv\Scripts\activate  # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run full pipeline
python3 src/met_fetch.py          # Collect data
python3 src/score_artworks.py     # Add happiness scores
python3 src/comprehensive_analysis.py  # Generate stats + figures