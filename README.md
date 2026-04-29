# New_Soundz

Personal Spotify analytics + artist recommendation engine. Hybrid CF using Last.fm 1K users as the backbone with my own listening history embedded as one extra user, trained with [LensKit](https://lkpy.lenskit.org/stable/).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync
```

This creates `.venv/` and installs everything. VS Code should auto-detect the interpreter; if not, point it at `.venv/bin/python`.

## Day 1: load streaming history

1. Drop your Spotify export files into `data/raw/`. Either format works:
   - **Account Data**: `StreamingHistory0.json`, `StreamingHistory1.json`, ...
   - **Extended Streaming History**: `Streaming_History_Audio_*.json`
   
   You can drop the whole unzipped export folder in there — the loader walks recursively.

2. Either run the notebook:
   ```bash
   uv run jupyter lab notebooks/01_explore_streams.ipynb
   ```
   
   ...or just run the loader as a script:
   ```bash
   uv run python -m spotify_recs.loader
   ```

3. Outputs land in `data/processed/`:
   - `all_streams.parquet` — every event
   - `real_plays.parquet` — plays ≥ 30s
   - `plays_by_artist.parquet` — recommender input
   - `plays_by_track.parquet` — analytics
   - `daily_listening.parquet` — dashboard

## Project structure

```
New_Soundz/
├── data/
│   ├── raw/          # Spotify JSON exports + Last.fm dataset (gitignored)
│   └── processed/    # Cleaned Parquet files (gitignored)
├── src/spotify_recs/ # Importable package
│   └── loader.py     # Day 1
├── notebooks/        # Exploration / scratch
├── app/              # Streamlit app (day 5)
└── reports/          # Quarto writeup (day 6)
```

## Day-by-day plan

- **Day 1** — Load + clean streaming history → Parquet
- **Day 2** — Pull Last.fm 1K dataset, fuzzy-match artists, build unified interaction matrix
- **Day 3** — LensKit pipeline (ALS / FlexMF), generate recommendations
- **Day 4** — Spotify API enrichment for recommended artists, build analytics queries
- **Day 5** — Streamlit app
- **Day 6** — Quarto writeup with evaluation
- **Day 7** — Buffer
