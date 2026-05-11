# New Soundz

A hybrid music recommender that surfaces under-listened artists from a user's Spotify history, with live controls to trade off popularity bias and recommendation diversity.

**Live demo:** https://new-soundz.streamlit.app

## What it does

You connect Spotify, the app pulls your top artists, and a recommender returns two ranked lists:

- **Classic recs**: catalog deep cuts surfaced by an ALS collaborative-filtering model trained on a 358K-user × 268K-artist dataset.
- **Modern recs**: post-2009 artists located via an external music-similarity graph, anchored to your classic recs as coordinate markers.

Two live sliders let you steer the output:

- **Popularity dampening (α)** divides each candidate's score by the norm of its embedding raised to α. At 0 the recommender favors popular artists; at 1 it goes popularity-blind.
- **MMR diversity (λ)** re-ranks the list to penalize candidates that cluster too closely to artists already picked. At 1 it's pure relevance; at 0.5 it deliberately spreads across clusters.

The Recommendations page also renders a UMAP cluster map (your seeds + recs jointly projected into 2D) and a force-directed artist similarity network.

## Architecture

The dataset powering the CF backbone is a 2008-09 snapshot — roughly half the artists a modern listener cares about don't exist in its vocabulary. The pipeline works around the gap in four stages:

1. **Routing**: incoming Spotify top artists are fuzzy-matched into the CF vocab. Unmatched names trigger proxy substitution: query the similarity graph for similar artists, intersect with the CF vocab, contribute weighted votes.
2. **Fold-in inference**: the seed vector solves a single matrix system against the trained ALS model. Sub-100ms per user.
3. **Reverse expansion**: top CF recs feed back into the similarity graph. Artists *not* in the CF vocab but co-endorsed by multiple classic recs surface as the modern-recs list.
4. **Content fallback**: for users whose seeds barely overlap the CF vocab, a genre-tag scorer takes over, using a multi-source genre cache (HuggingFace Spotify Tracks dataset + crowdsourced tags + Spotify API).

## Tech stack

Python · ALS collaborative filtering · Spotipy · SQLite (artist metadata cache) · Streamlit · UMAP · Plotly · NetworkX · Deployed to Streamlit Cloud

## Repo structure

```
src/spotify_recs/   # core package (loader, align, recommender, routing, cache, ...)
app/                # Streamlit app + privacy / terms pages
data/processed/     # runtime artifacts (artist name table, genre cache, sqlite)
models/             # ALS pickle (off-repo, lazy-downloaded at boot)
reports/            # project writeup
DEPLOY.md           # Streamlit Cloud deploy runbook
```

## Local setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync
uv run streamlit run app/main.py --server.port=8888
```

Local dev needs `.env` with `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `SPOTIFY_REDIRECT_URI`, `LASTFM_API_KEY` (and optionally `SPOTIFY_REFRESH_TOKEN` to boot straight into demo mode).

## Limits worth naming

- The ALS embeddings are frozen and the model doesn't learn from feedback. Same seed input always produces the same output (modulo the diversity slider).
- The CF vocabulary caps at 2008-09; modern picks come from the similarity-graph walk, which has its own crowdsourced quality variance.
- Spotify's API doesn't expose per-track play counts or listening time, so listening-history analytics are limited to what `/me/top/*` returns.
- The app is in Spotify Developer Mode (5-user allowlist); a server-side refresh token powers a public demo viewing the developer's data without requiring OAuth.
