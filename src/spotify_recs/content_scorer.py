"""Content-based fallback recommender.

Used when the CF fold-in path can't produce trustworthy recommendations —
the user's seed list yielded too few CF-vocab matches (`< 5`) so the user
vector is too sparse for ALS to score reliably.

The fallback ranks candidate CF artists by genre overlap with the user's
seed-derived genre profile. Both sides come from `cache.get_merged_genres`,
which unions HF Spotify Tracks dataset + Last.fm tags (+ optional Spotify
API genres). See `cache.py` for the merge logic.

Scoring: user genre profile is a `dict[genre, weight]`; candidate score is
the dot product between the user profile and the candidate's genre weights
(uniform 1.0 per genre, since candidate-side counts are mostly noise after
HF's playlist-derivation).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from spotify_recs.align import normalize_artist
from spotify_recs.cache import ArtistCache
from spotify_recs.recommender import (
    ARTIST_LOOKUP_PATH,
    JUNK_NORM_NAMES,
    is_denylisted,
    load_pipeline,
)


def build_genre_profile(
    seeds: list[tuple[str, float]],
    cache: ArtistCache,
    top_k_per_seed: int = 6,
) -> dict[str, float]:
    """Aggregate seed artists into a single user genre profile.

    `seeds`: [(artist_name, listen_weight), ...]. Listen weight comes from
    the route-time time-range scaling (short_term=2.0, medium_term=1.0).
    Weights flow to genres via the seed's top-K merged genres so a seed's
    most-confident genres dominate.
    """
    profile: dict[str, float] = {}
    for name, w in seeds:
        genres = cache.get_merged_genres(name)[:top_k_per_seed]
        if not genres:
            continue
        # Linear decay across the top-K so the seed's primary genre carries
        # the most weight: rank 0 -> 1.0, rank 1 -> ~0.83, etc.
        for rank, g in enumerate(genres):
            decay = (top_k_per_seed - rank) / top_k_per_seed
            profile[g] = profile.get(g, 0.0) + w * decay
    return profile


def top_popular_cf_ids(n: int = 1500) -> list[int]:
    """Top-N CF artists by ALS embedding norm. Norm correlates with
    popularity in implicit ALS, so this gives us a popularity-ranked
    candidate pool that excludes junk artists.
    """
    pipe = load_pipeline()
    scorer = pipe.node("scorer").component
    embeddings = scorer.item_embeddings
    item_ids = scorer.items.ids()

    norms = np.linalg.norm(embeddings, axis=1)
    top_idx = np.argsort(norms)[-n:][::-1]
    candidate_ids = [int(item_ids[i]) for i in top_idx]

    lookup = pd.read_parquet(ARTIST_LOOKUP_PATH)
    name_by_id = dict(zip(lookup["artist_id"].astype(int), lookup["canonical_name"]))
    return [
        aid for aid in candidate_ids
        if name_by_id.get(aid, "").lower() not in JUNK_NORM_NAMES
    ]


def content_recommend(
    seed_genres: dict[str, float],
    candidate_ids: list[int],
    cache: ArtistCache,
    n: int = 20,
    exclude_ids: set[int] | None = None,
    skip_uncached: bool = True,
) -> pd.DataFrame:
    """Score candidates by genre overlap with `seed_genres`. Top-N.

    If `skip_uncached` is True, candidates with no merged-genres row in the
    cache are skipped rather than triggering a Last.fm API hit. This keeps
    the first-call latency bounded; pre-warm the cache via the build script
    for production-quality fallback.

    Returns a DataFrame with [item_id, canonical_name, score] matching the
    shape of the CF `classic` recs so the UI can render either uniformly.
    """
    if not seed_genres:
        return pd.DataFrame(columns=["item_id", "canonical_name", "score"])

    exclude_ids = exclude_ids or set()
    lookup = pd.read_parquet(ARTIST_LOOKUP_PATH)
    name_by_id = dict(zip(lookup["artist_id"].astype(int), lookup["canonical_name"]))

    rows: list[tuple[int, str, float]] = []
    for aid in candidate_ids:
        if aid in exclude_ids:
            continue
        name = name_by_id.get(aid)
        if not name:
            continue
        if is_denylisted(name):
            continue

        if skip_uncached:
            row = cache._row(normalize_artist(name))
            if not row or row["genres_merged"] is None:
                continue
            cand_genres = list(json.loads(row["genres_merged"]))
        else:
            cand_genres = cache.get_merged_genres(name)

        if not cand_genres:
            continue

        score = sum(seed_genres.get(g, 0.0) for g in cand_genres)
        if score <= 0:
            continue
        rows.append((aid, name, score))

    if not rows:
        return pd.DataFrame(columns=["item_id", "canonical_name", "score"])

    df = pd.DataFrame(rows, columns=["item_id", "canonical_name", "score"])
    df = df.sort_values("score", ascending=False).head(n).reset_index(drop=True)
    return df


def prewarm_merged_genres(n: int = 1500) -> None:
    """Populate `genres_merged` for the top-N popular CF artists.

    One-time offline job. Without it, the content fallback (with its default
    `skip_uncached=True`) returns no candidates because the cache is empty.

    Last.fm rate limits us to 5 req/sec, so this takes ~5 minutes for 1500
    artists. Subsequent runs are SQLite reads.
    """
    import time

    candidate_ids = top_popular_cf_ids(n=n)
    lookup = pd.read_parquet(ARTIST_LOOKUP_PATH)
    name_by_id = dict(zip(lookup["artist_id"].astype(int), lookup["canonical_name"]))

    populated = 0
    skipped = 0
    started = time.time()
    with ArtistCache() as cache:
        for i, aid in enumerate(candidate_ids, 1):
            name = name_by_id.get(aid)
            if not name:
                continue
            row = cache._row(normalize_artist(name))
            if row and row["genres_merged"] is not None:
                skipped += 1
                continue
            try:
                cache.get_merged_genres(name)
                populated += 1
            except Exception as e:
                print(f"  failed {name}: {e}")
            if i % 100 == 0:
                elapsed = time.time() - started
                eta = elapsed / i * (len(candidate_ids) - i)
                print(
                    f"  {i}/{len(candidate_ids)} "
                    f"({populated} populated, {skipped} already cached) "
                    f"elapsed {elapsed:.0f}s, eta {eta:.0f}s"
                )
    print(f"\nDone: populated {populated}, skipped {skipped} already-cached.")


if __name__ == "__main__":
    import sys

    if "--prewarm" in sys.argv:
        n = 1500
        for arg in sys.argv:
            if arg.startswith("--n="):
                n = int(arg.split("=", 1)[1])
        prewarm_merged_genres(n=n)
    else:
        print("Usage: uv run python -m spotify_recs.content_scorer --prewarm [--n=1500]")
