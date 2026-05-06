# Final project steps

The recommender and Streamlit app are functionally done. Everything below is wrapper/deployment work. Total realistic effort to "portfolio-viewable on the internet": **~1 day** of focused work. Polish items are optional v2.

## Lane A — Ship it (critical path, ~1 day)

### 1. Demo mode wiring (~1 hr)
- New helper `_get_demo_client(refresh_token)` in `app/main.py` — builds a `spotipy.Spotify` client from a stored refresh token instead of running the OAuth flow.
- Session-state flag `mode = "demo" | "personal"`. Default to demo.
- "Connect your own Spotify" button at top of sidebar that flips to personal and triggers the existing OAuth path.
- Visible "DEMO — viewing rqhq's data" banner when in demo mode.
- Test locally first: dump your refresh token to `.env` as `SPOTIFY_REFRESH_TOKEN=...`, restart, confirm app loads as you with no OAuth click.

### 2. Off-repo ALS pickle hosting (~1 hr)
The `models/als.pkl` is 388MB — too big for Streamlit Cloud repo. Pick one:
- **HuggingFace Hub** (recommended): `huggingface_hub.hf_hub_download(repo_id, "als.pkl")`. Free, fast, public.
- **S3 / R2**: any object store with a public URL.
Wrap the download in `@st.cache_resource` so it runs once per server lifetime, persists in tmpfs.

### 3. Deploy to Streamlit Cloud (~1 hr)
- Push repo to GitHub (`models/` is already gitignored).
- Connect Streamlit Cloud, point at `app/main.py`, set secrets:
  - `SPOTIFY_CLIENT_ID`
  - `SPOTIFY_CLIENT_SECRET`
  - `SPOTIFY_REDIRECT_URI` = `https://<your-app>.streamlit.app/callback`
  - `SPOTIFY_REFRESH_TOKEN` (yours, for demo mode)
  - `LASTFM_API_KEY`
- Register the prod redirect URI in the Spotify dev dashboard (keep the local one too — Spotify allows multiple).
- First deploy will be slow: pickle download + load + UMAP fit. Watch logs, tune the "spinning up..." UX as needed.

### 4. Privacy policy + ToS pages (~half day, mostly content)
Required by Spotify dashboard review. Demo-only flow has a simpler privacy story (only your data). Can be:
- A single extra Streamlit page (`pages/privacy.py`), or
- Static HTML hosted anywhere with public URLs

### 5. Screencast / demo video (~1 hr)
~2-3 minute walkthrough: hero header, top artists, recs, UMAP, network. Fallback when live app is slow or down. Screen recording with QuickTime is fine.

## Lane B — Polish

### 6. ~~Multi-source genre enrichment + content fallback scorer~~ ✅ MOSTLY SHIPPED
- ✅ HF Spotify Tracks dataset → `hf_genres.py` (29.4k artists, 114 genres). Persisted lookup at `data/processed/hf_artist_genres.parquet`.
- ✅ Multi-source merge in `cache.get_merged_genres` (HF + Last.fm tags + optional Spotify API). Tag normalization + dedup + denoising (HF count-1 entries dropped when stronger signal exists).
- ✅ Content fallback scorer in `content_scorer.py`, wired into the sparse branch of `_compute_recs`. Validated on synthetic HYUKOH + Malcolm Todd + Frank Ocean profile → clean indie-rock recs.
- ✅ Prewarm CLI: `uv run python -m spotify_recs.content_scorer --prewarm --n=1500` (~5 min one-time).
- ❌ Genre coloring on UMAP / similarity network — tried, reverted. The 114→12 bucket map (`genre_buckets.py`) misclassifies enough artists at the rendered scale (Mos Def→electronic via 'trip hop', Snoop→funk/disco, etc.) that the colors mislead more than help. Module kept around for future retry with better mapping.

### 7. ~~Plays/minutes per top track in demo mode~~ ❌ DEFERRED INDEFINITELY
- Tried, pulled. The `plays_by_track.parquet` is rqhq's pre-Nov-2024 streaming export; joining it onto Spotify's `/me/top/tracks` for non-rqhq users produces misleading rows (one accidental track-name overlap shows rqhq's plays on someone else's listing). Only safe to bring back inside demo mode (Lane A #1) framing.
- Spotify's API doesn't expose play counts — no API path makes this work for arbitrary users.

### 7b. Recommender tuning sliders ✅ SHIPPED (this wasn't on the original Lane B list)
- α slider (`popularity_alpha`, 0–1): divides each candidate's ALS score by `‖item_emb‖^α`. 0 = raw ALS, 1 = popularity-blind.
- λ slider (`mmr_lambda`, 0.5–1.0): greedy MMR re-ranking using ALS-cosine for similarity. <1.0 trades relevance for cluster diversity.
- Both surfaced as live controls on the Recommendations page; cache key includes them. These now ARE the user-facing thesis ("New Soundz" = anti-popularity + anti-clustering).

### 8. Live ticker on now-playing strip ✅ SHIPPED
- `st.fragment(run_every="2s")` on `_now_playing_strip`. Progress bar + `m:ss / m:ss` caption tick without page rerun.
- Track label "All time" → "Past year" (Spotify's `long_term` is officially "several years" but in practice ~1 year).

### 9. Quarto portfolio page
- ✅ Project summary draft at [reports/project-summary.md](reports/project-summary.md). ~900 words, structured for portfolio readers.
- Lead is "constraint-led design": Last.fm 360K's 2008-09 vintage forced the classic+modern architecture, which then became the user-facing thesis ("expose listeners to under-listened music — whether 1972 deep cuts or 2024 indie").
- Honest limits section calls out frozen ALS, capped vocab, Last.fm graph quality, no per-user playcounts.
- TODO: convert to Quarto + add screenshots (UMAP makes the strongest hero image), publish on personal portfolio site.

## Cuts (don't bother)

- **Spotify Extension review for prod access.** 2-6 week wait. Allowlist + dev mode is fine for portfolio purposes; recruiters won't try to log in.
- **Anything requiring full streaming history for arbitrary users.** Export takes 30 days for them to receive — kills the use case.

## Known deployment gotchas (from CLAUDE.md, don't relearn the hard way)

- Spotify dashboard rejects `http://localhost`, accepts `http://127.0.0.1:<port>/<path>`. Use IP literal locally.
- Browser caches OAuth state in cookies — when changing OAuth flow type or app config, test in incognito.
- Spotify API quota is per-app, not per-user. All visitors share rqhq's quota in demo mode (fine for portfolio traffic, with caching).
- Sqlite cache: multi-reader fine, writes serialize. Cache misses rare post-warmup.
- Cold-start latency on Streamlit Cloud: first visitor pays ~30-60s. Show a clear loading state, not a blank screen.
- `.streamlit/config.toml` changes need a hard restart to take effect.
- Streamlit hot-reload re-runs `main.py` but does NOT re-import already-loaded modules from `src/spotify_recs/`. Module changes require Ctrl-C + restart.

## Suggested order tomorrow

1. **Coffee.**
2. Wire demo mode locally first (item 1) — confirm it works before touching deployment.
3. Move the pickle off-repo (item 2).
4. Push + deploy (item 3) — expect one frustrating hour of redirect-URI / cold-start debugging.
5. Privacy + ToS pages (item 4) — content work, low cognitive load, good end-of-day task.
6. Screencast (item 5) — only after all of the above is solid.

If everything ships, you can start Lane B tomorrow afternoon or just call it done and move to the Quarto writeup.
