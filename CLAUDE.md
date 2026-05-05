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

## Repo layout

```
src/spotify_recs/
  loader.py       # Day 1: Spotify export → parquet (DONE)
  align.py        # Day 2: Last.fm 360K + artist matching (DONE)
  recommender.py  # Day 3: ALS train + fold-in inference (DONE)
  lastfm_api.py   # Day 3: Last.fm API client w/ rate limit + tag denylist (DONE)
  cache.py        # Day 3: SQLite artist metadata cache, lazy-populated (DONE)
notebooks/        # Exploration
app/              # Streamlit (Day 5)
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
- 🚧 Day 4 in progress:
  - ✅ Spotify OAuth (Spotipy PKCE; redirect `http://127.0.0.1:5000/callback`; macOS AirPlay Receiver must be off or port reassigned)
  - ✅ Match-or-proxy router (`routing.py::route_artists`, threshold 95, decay 0.3)
  - ✅ Reverse-proxy expansion (`routing.py::expand_to_modern`) — surfaces post-2009 artists by walking Last.fm similars from CF recs. Live test: 100% input coverage on rqhq's account, ~75% of "modern picks" are post-2009 contemporary names per user judgment.
  - ✅ End-to-end CLI rig (`run_demo.py`) wires auth → fetch → route → fold-in → reverse-proxy → two ranked lists.
  - ⏭ Multi-source genre enrichment + content fallback scorer for sparse vectors (low priority — rqhq's account never triggers it, but still wanted for the 24 other allowlisted users).
- Day 5: Streamlit app (analytics + recs + demo mode). Open question: how to merge "classic" + "modern" rec lists into one focused output instead of two separate lists. User wants to narrow output count.
- Day 6: Deploy + privacy policy + Spotify dev dashboard config + screencast.

## Working preferences

- Push back when something seems off — user prefers a 2-min argument over rebuilding a day's work.
- Ask before destructive/sweeping changes (large refactors, deleting files, big dep changes). Don't ask for routine writes/tests/typo fixes.
- Stop after task completion; don't auto-continue to the next day.
- LensKit had major API changes in 2025.1; check current docs at https://lkpy.lenskit.org/stable/ before writing pipeline code — model knowledge from before 2025.1 is unreliable.
