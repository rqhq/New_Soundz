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
- **HF Spotify Tracks dataset is playlist-derived and noisy at single-track granularity.** Each track has one `track_genre` (the playlist it was scraped from), so an artist with one track on a "soul" playlist gets tagged soul even if they're hip-hop. Mitigation in `cache.get_merged_genres`: drop HF entries with count==1 when the artist has any tag with count≥3. Artists with only single-count entries (small catalogs) pass through unfiltered.
- **`plays_by_track.parquet` is single-user data (rqhq's pre-Nov-2024 streaming export).** Joining it onto Spotify's live `/me/top/tracks` for non-rqhq users produces misleading results — even one accidental match (e.g., friend-of-rqhq listening to the same song) shows rqhq's play counts on someone else's row. Feature was attempted, then pulled. Only safe to bring back inside demo mode (Lane A #1) where every viewer is explicitly framed as "viewing rqhq's data."
- **Genre coloring on UMAP/network was tried and reverted.** The 114-genre HF taxonomy collapses too cleanly to ~12 buckets, but bucket misclassifications (Mos Def→electronic via 'trip hop', Snoop→funk/disco via 'funk', JPEGMAFIA→metal via 'industrial' before weak-tag fallback) produced too many visible errors at the rendered scale. `genre_buckets.py` still exists with a weak-tag-fallback design for future retry; just not wired in. The viz reverted to type-based coloring (green seeds / orange star classics / cyan diamond modern).
- **`st.fragment(run_every="2s")` powers the now-playing live ticker.** Re-runs only the fragment, not the page; safe-ish on Spotify's API quota for personal-portfolio traffic. Requires Streamlit ≥1.37 (we're on 1.57).
- **Spotify `long_term` is officially "several years" but in practice ~1 year.** Spotify never announced the change, but community evidence is consistent. UI label was renamed "All time" → "Past year" to be honest.

## Repo layout

```
src/spotify_recs/
  loader.py       # Day 1: Spotify export → parquet (DONE)
  align.py        # Day 2: Last.fm 360K + artist matching (DONE)
  recommender.py  # Day 3: ALS train + fold-in inference (DONE)
  lastfm_api.py   # Day 3: Last.fm API client w/ rate limit + tag denylist (DONE)
  cache.py        # Day 3: SQLite artist metadata cache, lazy-populated (DONE)
  routing.py      # Day 4: match-or-proxy router + reverse-proxy expand_to_modern (DONE)
  hf_genres.py    # Day 5: HF Spotify Tracks dataset → artist→[(genre,count)] lookup (DONE)
  content_scorer.py  # Day 5: content fallback for sparse users + cache prewarm CLI (DONE)
  genre_buckets.py   # Day 5: 114→12 high-level bucket map (UNUSED — kept for future)
  get_demo_refresh_token.py  # Day 6: one-off OAuth CLI that writes SPOTIFY_REFRESH_TOKEN to .env (DONE)
notebooks/        # Exploration
app/
  main.py         # Day 5/6: Streamlit entrypoint — Overview / Analytics / Recommendations + demo-mode banner
  pages/
    1_Privacy_Policy.py    # Day 6: required by Spotify dashboard review
    2_Terms_of_Service.py  # Day 6: required by Spotify dashboard review
.streamlit/
  config.toml     # Dark Spotify theme. NO server.port — Streamlit Cloud needs the default 8501
reports/
  project-summary.md  # Portfolio writeup draft (Quarto-ready markdown)
DEPLOY.md         # Day 6: step-by-step Streamlit Cloud deploy runbook
models/           # als.pkl (370MB) — gitignored; hosted as GitHub Release asset, lazy-downloaded at boot
data/raw/         # Spotify exports + lastfm_360k.parquet (gitignored, training only)
data/processed/   # 3 small files force-added for runtime: artist_lookup.parquet, artist_cache.sqlite, hf_artist_genres.parquet. interactions.parquet is gitignored (training only).
```

Workflow: `uv run python -m spotify_recs.<module>`. Always `uv run`, never bare python.

## LensKit 2026.1.0 specifics (verified by inspecting installed source)

- `from_interactions_df(df)` builds a `Dataset` — auto-detects `user_id`/`item_id` columns. Pass `rating_col="count"` if needed; ImplicitMFScorer treats it as confidence weighting via `c_ui = 1 + alpha * count`.
- `ImplicitMFScorer` config defaults: `embedding_size=64, epochs=10, regularization=0.1, weight=40` (alpha). `use_ratings=False` is the right setting for implicit data.
- Pipeline build: `pipe = topn_pipeline(scorer, n=200); pipe.train(ds)`.
- **Fold-in cold-start path**: `RecQuery(history_items=ItemList(item_ids=[...], score=[weights...]))`, then `recommend(pipeline, query, n=20)`. The `history_items` interface is what powers all new-user inference — no manual `new_user_embedding` call needed.
- `RecQuery` API changed in 2026.1: arguments are keyword-only, the historical `user_items` attribute was removed. Don't trust pre-2026.1 examples.
- **`recommend_for_history` now takes two re-ranking knobs:**
  - `popularity_alpha` (0–1) divides each candidate's score by `‖item_emb‖^alpha`. 0 = raw ALS (popularity-biased), 1 = pure cosine. Pulls 200 candidates from the pipeline (instead of `n+buffer`) so re-ranking has room to surface buried items.
  - `mmr_lambda` (0.5–1.0) applies greedy Maximal Marginal Relevance over the dampened scores using ALS-cosine for similarity. 1.0 = no diversity step. <1.0 trades relevance for cluster spread. Implementation in `_mmr_rerank`.
  - Both surfaced as live sliders on the Recommendations page; cache key includes both so each combo recomputes fresh.

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
- ✅ Day 5 (`app/main.py`):
  - ✅ **Overview page**: hero header (avatar + display name + follower count), now-playing strip, top-8 artists as a row of large clickable album-art cards.
  - ✅ **Analytics page**: KPI row, top genres horizontal bar (Spotify-green Plotly), artist movement table (4w vs 6mo rank with ▲/▼/🆕 markers), top tracks per time range with album thumbnails + popularity progress bars. Tab labels: "Last 4 weeks / Last 6 months / Past year" (renamed from "All time"). Listening-clock plot was tried with polar bars and scrapped — the live API's 50-play window is too sparse for it to look good; revisit only when demo mode brings full export.
  - ✅ **Recommendations page**: Mixed/Classic/Modern toggle on rec list, **α/λ tuning sliders** (popularity dampening + MMR diversity, both wired into `recommend_for_history`), **UMAP cluster map** (joint `fit_transform` over 1500 popular CF artists + user seeds + classic recs, modern recs excluded since they have no embedding), **artist similarity network** (force-directed via networkx, ALS-cosine edges within CF + Last.fm support edges modern→CF, glowing-halo markers, labels capped to top-18 by score). Network feeds on top-50 classic recs (`classic_expanded`) so modern recs always anchor; rec list still shows top-20. UMAP and network markers color by *type* only (seed/classic/modern); genre coloring was tried and reverted (see gotchas).
  - ✅ **Live now-playing ticker** via `st.fragment(run_every="2s")` — progress bar + `m:ss / m:ss` caption update without page rerun.
  - ✅ Dark Spotify theme via `.streamlit/config.toml` (#1DB954 primary, #121212 base).
  - ✅ Multi-source genre enrichment **data layer** (`hf_genres.py` + `cache.get_merged_genres`) — HF Spotify Tracks dataset (29.4k artists, 114 genres) + Last.fm tags + optional Spotify API genres, normalized + deduped + filtered. 1500-artist prewarm CLI lives in `content_scorer.py --prewarm`.
  - ✅ Content-based fallback scorer (`content_scorer.py`) — wired into the sparse branch of `_compute_recs`. Synthetic HYUKOH+Malcolm Todd+Frank Ocean test produced clean indie-rock recs (Beck, Modest Mouse, Phoenix, MGMT). Won't trigger for rqhq but does for unmatched-heavy users.
  - ⏭ **Plays/minutes per top track — pulled** after testing on friend's account: `plays_by_track.parquet` is rqhq-only data; even rare accidental matches show rqhq's plays on someone else's row. Only feasible inside demo mode framing.
  - 🤔 Open question still alive: still no single merged rec list — Mixed-toggle (round-robin interleave) is the current answer, may revisit for finer control.
- ✅ Project summary writeup drafted at [reports/project-summary.md](reports/project-summary.md). ~900 words, portfolio-ready prose; lead is "constraint-led design" (Last.fm 360K's age forced the two-tier classic+modern architecture, which then became the user-facing thesis).
- New runtime deps from Day 5: `umap-learn` (pulls numba+llvmlite, ~40MB), `networkx` (already present via scipy chain). HF dataset adds `data/raw/hf_spotify_tracks.csv` (~20MB) + `data/processed/hf_artist_genres.parquet`.
- ✅ Day 6 (deployed):
  - ✅ **Demo mode** wired into [app/main.py](app/main.py): `_get_demo_client(refresh_token)` builds a Spotify client from a server-side refresh token (no OAuth click). Session-state `mode` toggle defaults to `"demo"` if `SPOTIFY_REFRESH_TOKEN` is set; sidebar offers "Connect your own Spotify" → `"personal"` and "← Back to demo" round-trip. Demo banner renders on every page in demo mode.
  - ✅ **Off-repo ALS pickle** via GitHub Release asset. `_ensure_als_pickle()` lazy-downloads `models/als.pkl` from `SPOTIFY_ALS_PICKLE_URL` on first boot (with a progress bar), cached via `@st.cache_resource`. Asset URL: `https://github.com/rqhq/New_Soundz/releases/download/v0.1/als.pkl`.
  - ✅ **Privacy + ToS pages** at `app/pages/` — auto-discovered by Streamlit, public URLs `/Privacy_Policy` and `/Terms_of_Service` registered with the Spotify dashboard.
  - ✅ **Editorial denylist** (`recommender.is_denylisted()`) — regex `\br[^a-z]{0,3}kelly\b` filters R. Kelly + collabs without false-positives on Kelly Clarkson / Kelly Rowland / Robert Kelly. Applied at all 3 rec output paths: `recommender._itemlist_to_df`, `routing.expand_to_modern`, `content_scorer.content_recommend`.
  - ✅ **Top-genres bar fixed** to use `cache.get_merged_genres` (HF + Last.fm + Spotify) instead of just `a["genres"]`. Spotify alone returns empty for ~22/50 of any modern listener's top artists; merged source gives meaningful counts (10-20 range vs 1-4 before).
  - ✅ **"New Soundz" wordmark** on the Overview page (gradient Spotify-green → mint), `page_title` updated.
  - ✅ **Live deploy** at `https://<rqhq's-subdomain>.streamlit.app` — first boot ~60-90s (pickle download + load + UMAP fit), subsequent boots fast.
  - ⏭ Screencast (deferred to next session).
  - ⏭ Spotify Extension review for production access (2-6 wk, accept dev-mode 25-user cap for now).

## Day 6 deployment lessons (don't relearn)

- **`.streamlit/config.toml` must NOT pin `server.port`** — Streamlit Cloud's health check probes the default 8501. Pinning to 8888 (the local-dev workaround for macOS AirPlay) caused the deploy's first boot to fail with `connection refused on 127.0.0.1:8501`. Local dev now needs `--server.port=8888` explicitly on the CLI.
- **`.spotify_token_cache.json` is contaminated by whoever last OAuth'd.** When testing on a friend's account first, then trying to swap to your own, the streamlit "Connect your own Spotify" button must `TOKEN_CACHE_PATH.unlink(missing_ok=True)` — otherwise it re-validates the friend's still-valid token and silently routes you back as them. Plus the Spotify-side browser cookie auto-logs you in as whoever's signed into accounts.spotify.com — incognito or `accounts.spotify.com/en/logout` first.
- **`get_demo_refresh_token.py`** bypasses both contamination paths: standalone OAuth via auth-code flow on a separate cache file, writes the resulting refresh token straight to `.env`. Use this when you need a clean refresh token for any account.
- **The 3 small data files (`artist_lookup.parquet`, `artist_cache.sqlite`, `hf_artist_genres.parquet`, ~9 MB total) are force-added** to the repo despite `data/processed/*` being gitignored. They're needed at runtime; only `interactions.parquet` (79 MB, training only) stays out.
- **Streamlit secrets are TOML.** Common breakage: missing quotes around the value, smart quotes from a doc app, line break inside a long token value. Refresh tokens have no special chars — if the TOML parser complains, it's quote/paste shape.
- **GitHub Release public download URLs work without auth** (`https://github.com/<owner>/<repo>/releases/download/<tag>/<asset>`) — `curl -IL` confirms 200 + `application/octet-stream`. No `gh` API token needed at runtime.
- **`@st.cache_resource` on `_load_recsys` means the pickle download happens once per worker lifetime**, not per session. Subsequent visitors hit a warm worker instantly.

## Working preferences

- Push back when something seems off — user prefers a 2-min argument over rebuilding a day's work.
- Ask before destructive/sweeping changes (large refactors, deleting files, big dep changes). Don't ask for routine writes/tests/typo fixes.
- Stop after task completion; don't auto-continue to the next day.
- LensKit had major API changes in 2025.1; check current docs at https://lkpy.lenskit.org/stable/ before writing pipeline code — model knowledge from before 2025.1 is unreliable.
