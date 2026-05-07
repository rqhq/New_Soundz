# Streamlit Cloud deploy runbook

One-time setup for the public demo deployment. Assumes you have:
- A GitHub repo for this project
- A Streamlit Cloud account (free tier is fine)
- The `gh` CLI installed and authenticated

## 0. Pre-flight (local sanity checks)

```bash
# Make sure the demo refresh token works
uv run python -m spotify_recs.get_demo_refresh_token   # only re-run if needed
uv run streamlit run app/main.py
# Open http://localhost:8888 — should boot into demo mode showing your data
```

## 1. Force-add runtime data files

These are gitignored by default but the deployed app needs them. They total ~9 MB.

```bash
git add -f \
  data/processed/artist_lookup.parquet \
  data/processed/artist_cache.sqlite \
  data/processed/hf_artist_genres.parquet
git commit -m "deploy: include runtime data artifacts"
```

Don't add `interactions.parquet` (79 MB) — that's training-only.

## 2. Host the ALS pickle as a GitHub Release asset

```bash
gh release create v0.1 models/als.pkl \
  --title "ALS model v0.1" \
  --notes "Trained LensKit ALS pipeline (LFM-360K, 64-dim)"
```

Grab the public asset URL — format is:
```
https://github.com/<owner>/<repo>/releases/download/v0.1/als.pkl
```

You can also get it via:
```bash
gh release view v0.1 --json assets -q '.assets[0].url'
```

## 3. Push to GitHub

```bash
git push -u origin main
```

## 4. Streamlit Cloud config

1. New app → point at this repo, `app/main.py`, branch `main`.
2. Advanced settings → set Python version to 3.13 (or whatever `uv.lock` pins).
3. Secrets — paste TOML below, fill in your values:

```toml
SPOTIFY_CLIENT_ID = "..."
SPOTIFY_CLIENT_SECRET = "..."
SPOTIFY_REDIRECT_URI = "https://<your-app>.streamlit.app/callback"
SPOTIFY_REFRESH_TOKEN = "..."   # from .env
LASTFM_API_KEY = "..."
SPOTIFY_ALS_PICKLE_URL = "https://github.com/<owner>/<repo>/releases/download/v0.1/als.pkl"
```

## 5. Spotify dashboard

Add the production redirect URI:
```
https://<your-app>.streamlit.app/callback
```

Keep `http://127.0.0.1:8888/callback` registered too — useful for local dev.

## 6. First-boot expectations

Cold start will be slow (~60-90s):
- ~30s downloading the 370 MB pickle
- ~10s loading it into memory
- ~10s fitting UMAP background on first Recommendations page hit
- ~5s priming any uncached Last.fm tags

The `_ensure_als_pickle` helper shows a progress bar while downloading. Subsequent visitors hit a warm worker — instant.

## 7. Common failure modes

- **`SPOTIFY_ALS_PICKLE_URL is not set`** — secrets aren't applied; redeploy.
- **`code_verifier was incorrect`** — Streamlit needs auth-code flow (already configured), not PKCE. If this shows up, check `_build_auth_manager` is using `SpotifyOAuth`.
- **`error=server_error`** on Spotify side — stale OAuth state in browser cookies. Test in incognito.
- **404 from the Release asset URL** — make sure the release is public (default for public repos).
- **Workers OOM** — Streamlit Cloud free tier is 1 GB. Pickle + UMAP background + cached pandas frames = ~700 MB. If you OOM, retrain ALS with `embedding_size=32` (halves the pickle).
