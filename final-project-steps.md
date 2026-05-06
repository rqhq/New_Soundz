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

## Lane B — Polish (optional, ~1-2 days, can ship as v2)

### 6. Multi-source genre enrichment + content fallback scorer
- The only Day 4 step that wasn't built. rqhq's account doesn't trigger the sparse fallback so it didn't block anything.
- Real value: unlocks **genre coloring on the UMAP and similarity network**. Right now those clusters are positional but visually uninterpretable beyond seed/rec/modern color tiers.
- Priority order from CLAUDE.md: HF Spotify Tracks dataset → Last.fm `getTopTags` → Spotify API genres → Wikidata.

### 7. Plays/minutes per top track in demo mode
- Easy add: gate on `mode == "demo"`, pull from your existing parquet (loader.py output), join to top tracks by track id.
- Mild interest payoff. Skip if pressed for time.

### 8. Quarto portfolio page
- Already in the plan. Project writeup links to deployed app + screencast.
- Mention the recommender architecture, the proxy substitution insight, the UMAP/network visualizations.

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
