# Project state — read this first

Personal Spotify analytics + artist recommendation engine. Portfolio piece for Data Science / ML co-op interviews. Solo dev (rquinlan3), uv-managed Python project.

## What this actually is now (not what the original prompt said)

- **Multi-user webapp**, not a personal-data dashboard. Users link their Spotify account via OAuth, see their own analytics, get their own recommendations.
- **Streamlit** frontend, deployed to Streamlit Cloud (planned). Spotify dev mode caps usage at 25 manually allowlisted users — accept this and lean on a "demo mode" + screencast for portfolio viewing. App is framed as "under development."
- **No Quarto writeup.** Project page goes on user's Quarto portfolio when finished.

## Architecture

### Recommender — hybrid CF with proxy substitution + content fallback

1. **CF backbone**: ALS trained on Last.fm 360K (one-time, offline). 358,872 users × 267,739 artists, density 0.0181%. LensKit 2026.1.0 pipeline API.
2. **Fold-in inference** for new users at request time (single matrix solve, sub-100ms). User vector built from Spotify Top Artists short_term (4w) + medium_term (6m), weighted toward short_term.
3. **Proxy substitution** for unmatched artists: query Last.fm API `artist.getSimilar`, intersect with CF vocab, contribute decayed weights to user vector. Decay = 0.3 starting point. **Two-hop substitution worth doing** when one-hop returns empty.
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
- **`pyarrow.parquet` needs explicit import** — `pa.parquet` won't work.
- **`implicit` package fails to build from source on Apple Silicon** with current dependencies. Already removed from pyproject.toml. LensKit's own ALS doesn't need it.
- **`uv sync` warning** about `tool.uv.dev-dependencies` deprecation — non-blocking, fix when convenient.

## Repo layout

```
src/spotify_recs/
  loader.py     # Day 1: Spotify export → parquet (DONE)
  align.py      # Day 2: Last.fm 360K + artist matching (DONE)
notebooks/      # Exploration
app/            # Streamlit (Day 5)
reports/        # currently empty, no Quarto
data/raw/       # Spotify exports + lastfm_360k.parquet (TSV deleted post-conversion)
data/processed/ # all_streams, real_plays, plays_by_artist, plays_by_track, daily_listening, interactions, artist_lookup, my_artist_matches
```

Workflow: `uv run python -m spotify_recs.<module>`. Always `uv run`, never bare python.

## Status / next up

- ✅ Day 1: streaming history loader. 103,142 events → 28,752 real plays (≥30s) → 3,678 artists. Year of data: 2024-11-16 → 2025-11-16. Skip rate 72%.
- ✅ Day 2: Last.fm 360K aligned. 50.6% play-weighted match coverage (the structural ceiling). interactions.parquet + artist_lookup.parquet written.
- ⏭ Day 3 (next): Train ALS on Last.fm 360K. Implement fold-in. Scaffold Last.fm API client + SQLite cache. Smoke-test `getSimilar` on a few unmatched artists end-to-end.
- Day 4: Spotify OAuth + multi-source genre enrichment + content fallback scorer.
- Day 5: Streamlit app (analytics + recs + demo mode).
- Day 6: Deploy + privacy policy + Spotify dev dashboard config + screencast.

## Working preferences

- Push back when something seems off — user prefers a 2-min argument over rebuilding a day's work.
- Ask before destructive/sweeping changes (large refactors, deleting files, big dep changes). Don't ask for routine writes/tests/typo fixes.
- Stop after task completion; don't auto-continue to the next day.
- LensKit had major API changes in 2025.1; check current docs at https://lkpy.lenskit.org/stable/ before writing pipeline code — model knowledge from before 2025.1 is unreliable.
