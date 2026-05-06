# Project state — read this first

Personal Spotify analytics + artist recommendation engine. Portfolio piece for Data Science / ML co-op interviews. Solo dev (rquinlan3), uv-managed Python project.

## What this actually is

- **Multi-user webapp** Users link their Spotify account via OAuth, see their own analytics, get their own recommendations.
- **Streamlit** frontend, deployed to Streamlit Cloud (planned). Spotify dev mode caps usage at 25 manually allowlisted users — accept this and lean on a "demo mode" + screencast for portfolio viewing. App is framed as "under development."
- **No Quarto writeup.** Project page goes on user's Quarto portfolio when finished.

## Architecture

### Recommender — hybrid CF with proxy substitution + content fallback

1. **CF backbone**: ALS trained on Last.fm 360K (one-time, offline). 358,872 users × 267,739 artists, density 0.0181%. LensKit 2026.1.0 pipeline API.
2. **Fold-in inference** for new users at request time (single matrix solve, sub-100ms). User vector built from Spotify Top Artists short_term (4w) + medium_term (6m), weighted toward short_term.
3. **Proxy substitution** for unmatched artists: query Last.fm API `artist.getSimilar`, intersect with CF vocab, contribute decayed weights to user vector. Decay = 0.3 starting point — applied as the `rating` field value on proxy entries while direct seeds sit at 1.0 (LensKit fold-in only honors weights when `use_ratings=True`; `recommend_for_history` flips this at inference). **Two-hop substitution worth doing** when one-hop returns empty.
4. **Content-based fallback scorer** kicks in when user vector is too sparse (CF can't be trusted). Scores by genre-tag overlap.
5. **Multi-source genre enrichment**, in priority order: HF Spotify Tracks dataset (cheapest) → Last.fm `artist.getTopTags` (NOT `getInfo` — getInfo only returns ~5) → Spotify API genres → Wikidata (only if 1-3 leave too many gaps). One unified SQLite cache table per artist; merged dedup'd genre list is the content vector.

### Spotify API constraints to remember

- **Audio features deprecated Nov 2024.** `/recommendations` and `/related-artists` also deprecated. Do not try to use them.
- Live API gives: Recently Played (last 50), Top Artists/Tracks (short/medium/long buckets, ≤50 each), Currently Playing, Library, Followed, Playlists. That's it.
- **Full streaming history is NOT available via API** — only via the manual data export (30-day wait). Don't design any flow that assumes it.
- Spotify genres are sparse and unreliable even for huge artists (Tyler, The Creator routinely empty). Never use as the sole genre source.

### Last.fm API
- Free, key required, 5 req/sec.
- Used for: `artist.getSimilar` (proxy substitution graph), `artist.getTopTags` (genre tags).
- Crowdsourced tags include junk (`seen live`, `USA`, year tags, mood words) — needs denylist + intersection with a known genre taxonomy (HF dataset's 125 genres works as the allowlist).

### Caching
SQLite. One row per artist with columns for each genre source separately + merged + similar_lastfm + last_updated. Lazy population on demand.

## Gotchas / constraints

- **Last.fm 360K is a 2008-2009 snapshot.** ~50% of any modern listener's plays will be for artists not in the dataset (Kendrick, Tyler, Frank Ocean, Anderson .Paak, Disclosure, Jamie xx, JPEGMAFIA, HYUKOH, Malcolm Todd, etc.). This is the structural ceiling, not a normalization bug — verified by direct lookup. Proxy substitution + content fallback are how we work around it.
- **Fuzzy artist matching uses `fuzz.ratio` at threshold 95.** `token_set_ratio` was a disaster (matched "Malcolm Todd" → "todd" at score 100). At 92 we still got "George Clanton" → "george clinton" which would misroute 248 plays of vaporwave to P-Funk. Don't lower the threshold without re-auditing.
- **Spotify API in dev mode caps at 25 hand-allowlisted users.** Submit Extension review for production access (2-6 weeks). Demo mode + screencast for portfolio.
- **Privacy policy + ToS pages are required** for Spotify OAuth approval.
- **Streamlit OAuth must use `SpotifyOAuth` (auth-code + client_secret), NOT `SpotifyPKCE`.** PKCE keeps the code_verifier in memory; when the browser navigates to Spotify and back, Streamlit can lose its session and a new auth_manager generates a fresh verifier — exchange fails with `code_verifier was incorrect`. Auth-code flow has no per-request secret to lose. CLI (`spotify_auth.py`) still uses PKCE because it has a stable in-process listener.
- **macOS AirPlay Receiver claims ports 5000 and 7000.** Either toggle off (System Settings → General → AirDrop & Handoff → AirPlay Receiver), or use a different port. Streamlit currently runs on 8888; CLI OAuth listener uses 5000 (requires AirPlay off).
- **Spotify dashboard rejects `http://localhost`** as not-secure but accepts `http://127.0.0.1:<port>/<path>`. Always use the IP literal.
- **Browser OAuth state caches in cookies.** When changing OAuth flow type (PKCE↔auth-code) or app config, the browser can replay stale OAuth state and trigger `error=server_error` on Spotify's side. Test in incognito or clear cookies for `accounts.spotify.com`.
- **`pyarrow.parquet` needs explicit import** — `pa.parquet` won't work.
- **`implicit` package fails to build from source on Apple Silicon** with current dependencies. Already removed from pyproject.toml. LensKit's own ALS doesn't need it.
- **`uv sync` warning** about `tool.uv.dev-dependencies` deprecation — non-blocking, fix when convenient.
- **`umap.UMAP.transform()` segfaults on Apple Silicon** with current numba/pynndescent versions. The Streamlit UMAP cluster map works around this by doing a single joint `fit_transform` over (background ∪ user seeds ∪ recs) per user, cached for an hour. Don't try to call `transform()` separately.
- **Streamlit `@st.cache_data` invalidates on the decorated function's source code, not on transitive callees.** When `routing.py::expand_to_modern` gained a `support_seeds` column, in-flight cached results from `_compute_recs` did NOT auto-invalidate because `_compute_recs`'s body was unchanged. The Log out button calls `st.cache_data.clear()` as a workaround; otherwise wait the 1hr TTL or manually clear.

## Repo layout

```
src/spotify_recs/
  loader.py       # Day 1: Spotify export → parquet (DONE)
  align.py        # Day 2: Last.fm 360K + artist matching (DONE)
  recommender.py  # Day 3: ALS train + fold-in inference (DONE)
  lastfm_api.py   # Day 3: Last.fm API client w/ rate limit + tag denylist (DONE)
  cache.py        # Day 3: SQLite artist metadata cache, lazy-populated (DONE)
  routing.py      # Day 4: match-or-proxy router + reverse-proxy expand_to_modern (DONE)
notebooks/        # Exploration
app/
  main.py         # Day 5: Streamlit entrypoint — Overview / Analytics / Recommendations pages
.streamlit/
  config.toml     # Dark Spotify theme + server.port=8888
reports/          # currently empty, no Quarto
models/           # als.pkl (388MB pickled trained pipeline)
data/raw/         # Spotify exports + lastfm_360k.parquet
data/processed/   # parquet outputs + artist_cache.sqlite
```

Workflow: `uv run python -m spotify_recs.<module>`. Always `uv run`, never bare python.

## LensKit 2026.1.0 specifics (verified by inspecting installed source)

- `from_interactions_df(df)` builds a `Dataset` — auto-detects `user_id`/`item_id` columns. Pass `rating_col="count"` if needed; ImplicitMFScorer treats it as confidence weighting via `c_ui = 1 + alpha * count`.
- `ImplicitMFScorer` config defaults: `embedding_size=64, epochs=10, regularization=0.1, weight=40` (alpha). `use_ratings=False` is the right setting for implicit data.
- Pipeline build: `pipe = topn_pipeline(scorer, n=200); pipe.train(ds)`.
- **Fold-in cold-start path**: `RecQuery(history_items=ItemList(item_ids=[...], score=[weights...]))`, then `recommend(pipeline, query, n=20)`. The `history_items` interface is what powers all new-user inference — no manual `new_user_embedding` call needed.
- `RecQuery` API changed in 2026.1: arguments are keyword-only, the historical `user_items` attribute was removed. Don't trust pre-2026.1 examples.

## Day 3 findings worth remembering

- **Junk artist filter list** lives in `recommender.JUNK_NORM_NAMES`. Catches `[unknown]`, `various artists`, `soundtrack`, etc. — these dominate ALS scores otherwise (2.4M+ plays each).
- **ALS popularity bias** in raw "me" recs (Smash Mouth, Katy Perry, Nickelback) but **fold-in recs are visibly cleaner** — focused 5-artist seed produced very coherent neighbors. This is the real-user runtime path so quality is fine; no fix needed.
- **Proxy substitution feasibility validated end-to-end**:
  - Strong (>15/50 proxies in CF vocab): Tyler, The Creator (17), Frank Ocean (17), Anderson .Paak (18), Kendrick Lamar (22)
  - Decent (~12/50): JPEGMAFIA
  - Weak (≤4/50): HYUKOH (Korean indie), Malcolm Todd (2024) — these are exactly the cases where two-hop + content fallback earn their keep
- **Last.fm tags work universally** — every test artist returned 6+ usable tags after denylist filtering. Content scorer has signal.
- **Tag normalization needed** before merging: case dedup (`Hip-Hop` vs `hip hop`).

## Status / next up

- ✅ Day 1: streaming history loader.
- ✅ Day 2: Last.fm 360K aligned, 50.6% play-weighted match coverage.
- ✅ Day 3: ALS trained (`models/als.pkl`), fold-in path tested, Last.fm cache scaffolded, proxy substitution validated end-to-end on 7 unmatched artists.
- ✅ Day 4:
  - ✅ Spotify OAuth (auth-code flow in Streamlit; PKCE in CLI). Redirect `http://127.0.0.1:8888/callback` for the app, `:5000/callback` for the CLI.
  - ✅ Match-or-proxy router (`routing.py::route_artists`, threshold 95, decay 0.3)
  - ✅ Reverse-proxy expansion (`routing.py::expand_to_modern`) — surfaces post-2009 artists by walking Last.fm similars from CF recs. Live test: 100% input coverage on rqhq's account, ~75% of "modern picks" are post-2009 contemporary names per user judgment. Now also returns a `support_seeds` column listing `(seed_canonical_name, sim_score)` pairs per modern rec — used by the similarity-network viz to draw modern→CF edges.
  - ✅ End-to-end CLI rig (`run_demo.py`) wires auth → fetch → route → fold-in → reverse-proxy → two ranked lists.
  - ⏭ Multi-source genre enrichment + content fallback scorer for sparse vectors (low priority — rqhq's account never triggers it, but still wanted for the 24 other allowlisted users). Note: this is *also* what would unlock genre-coloring on the UMAP and similarity-network plots; right now their clusters are positional but visually uninterpretable beyond seed/rec/modern color tiers.
- 🚧 Day 5 in progress (`app/main.py`):
  - ✅ **Overview page**: hero header (avatar + display name + follower count), now-playing strip, top-8 artists as a row of large clickable album-art cards.
  - ✅ **Analytics page**: KPI row, top genres horizontal bar (Spotify-green Plotly), artist movement table (4w vs 6mo rank with ▲/▼/🆕 markers), top tracks per time range with album thumbnails + popularity progress bars. Listening-clock plot was tried with polar bars and scrapped — the live API's 50-play window is too sparse for it to look good; revisit only when demo mode brings full export.
  - ✅ **Recommendations page**: Mixed/Classic/Modern toggle on rec list, **UMAP cluster map** (joint `fit_transform` over 1500 popular CF artists + user seeds + classic recs, modern recs excluded since they have no embedding), **artist similarity network** (force-directed via networkx, ALS-cosine edges within CF + Last.fm support edges modern→CF, glowing-halo markers, labels capped to top-18 by score). Network feeds on top-50 classic recs (`classic_expanded`) so modern recs always anchor; rec list still shows top-20.
  - ✅ Dark Spotify theme via `.streamlit/config.toml` (#1DB954 primary, #121212 base).
  - ⏭ Plays/minutes per top track — deferred to demo mode (Spotify API exposes ranking but not counts; only the export has them).
  - ⏭ Demo mode (Step 4): seed Streamlit from rqhq's exported `StreamingHistory*.json` so unlocked plots (circadian, taste-drift, plays/minutes per track) become viewable without needing each visitor to upload their own.
  - 🤔 Open question still alive: still no single merged rec list — Mixed-toggle (round-robin interleave) is the current answer, may revisit for finer control.
- New runtime deps from Day 5: `umap-learn` (pulls numba+llvmlite, ~40MB), `networkx` (already present via scipy chain).
- Day 6: Deploy + privacy policy + Spotify dev dashboard config + screencast. See "Demo-mode deployment plan" below.

## Demo-mode deployment plan (Day 6)

Goal: a public URL where unallowlisted visitors immediately see rqhq's analytics/recs/network without going through OAuth. Allowlisted users (the other 24) can optionally flip to "Connect your own Spotify."

### Architecture
- **Server-side stored refresh token** for rqhq. OAuth happens once locally → refresh token persisted in Streamlit Cloud secrets (`SPOTIFY_REFRESH_TOKEN`). At app boot, hydrate a Spotipy client from it; Spotipy auto-refreshes the access token when it expires.
- **Two modes, one app.** Default `mode="demo"` (always-as-rqhq). A "Connect your Spotify" button flips `mode="personal"` and triggers the existing OAuth flow. Stored in `st.session_state`.
- **Code change is small** — ~50 lines: `_get_demo_client(refresh_token)`, the mode toggle, a "DEMO — viewing rqhq's data" banner.

### Deployment gotchas
- **The 388MB ALS pickle is the biggest problem.** Streamlit Cloud free tier has repo size + slow cold-start constraints. Options, in order of preference:
  1. Host on HuggingFace Hub or S3, download to a temp dir at boot, cache via `@st.cache_resource`. Works around repo-size limit.
  2. Git LFS (eats LFS quota).
  3. Retrain with `embedding_size=32` (halves the pickle to ~190MB) — last resort, hurts rec quality.
- **Cold-start latency**: first visitor pays for pickle download (10-30s) + load (5-10s) + UMAP fit (10s) + first Last.fm cache misses. Need an explicit "spinning up..." UX, not a blank screen.
- **Spotify redirect URI must match exactly.** Register `https://<app>.streamlit.app/callback` in the dashboard for prod. Keep `http://127.0.0.1:8888/callback` for local dev (Spotify allows multiple).
- **Privacy policy + ToS pages required** for Spotify dashboard review. Demo-only flow has a simpler privacy story (only your data) but the pages are still mandatory.
- **All visitors share rqhq's Spotify API quota** (~hundreds req/min app-wide). With `@st.cache_data` aggressively warming, this is fine for portfolio-tier traffic.
- **Sqlite cache concurrency**: multi-reader is fine, writes serialize. Cache misses are rare post-warmup — won't bite at low traffic.

### Suggested attack order
1. Add `_get_demo_client(refresh_token)` + `mode` toggle, test locally with refresh token in `.env` (~1 hr).
2. Move ALS pickle off-repo, add lazy-download path (~1 hr).
3. Deploy to Streamlit Cloud, register prod redirect URI, end-to-end run (~1 hr).
4. Privacy policy + ToS + screencast (~half day, mostly content).

## Working preferences

- Push back when something seems off — user prefers a 2-min argument over rebuilding a day's work.
- Ask before destructive/sweeping changes (large refactors, deleting files, big dep changes). Don't ask for routine writes/tests/typo fixes.
- Stop after task completion; don't auto-continue to the next day.
- LensKit had major API changes in 2025.1; check current docs at https://lkpy.lenskit.org/stable/ before writing pipeline code — model knowledge from before 2025.1 is unreliable.
