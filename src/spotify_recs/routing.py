"""Match-or-proxy router.

Given a list of Spotify artists, route each one into the CF vocab as either:
  - a DIRECT seed (exact normalized match, or rapidfuzz match >= threshold)
  - one or more PROXY contributions (Last.fm similar artists that ARE in vocab)
  - unmatched (couldn't help — surfaced as content-fallback signal)

Returns the weighted dict shape that `recommend_for_history` expects.

Decisions baked in (override at call site if needed):
  - fuzz threshold 95 (lower is dangerous; see CLAUDE.md gotchas)
  - proxy decay 0.3, applied as a multiplicative factor on the Last.fm
    similarity score
  - time_range_weight = 2.0 for short_term, 1.0 for medium_term

The weights returned are passed verbatim as the `rating` field on LensKit's
ItemList; `recommend_for_history` flips `use_ratings=True` so they're honored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd
from rapidfuzz import fuzz, process

from spotify_recs.align import normalize_artist
from spotify_recs.cache import ArtistCache
from spotify_recs.recommender import ARTIST_LOOKUP_PATH, is_denylisted

PROXY_DECAY_DEFAULT = 0.3
FUZZY_THRESHOLD_DEFAULT = 95
SHORT_TERM_WEIGHT = 2.0
MEDIUM_TERM_WEIGHT = 1.0


@dataclass
class RoutingResult:
    weights: dict[int, float]                # CF artist_id -> rating
    routing: list[dict]                      # per-input-artist diagnostics
    unmatched: list[str]                     # Spotify names with no direct or proxy hit
    n_direct: int
    n_proxy_only: int


def build_norm_to_id(lookup: pd.DataFrame | None = None) -> dict[str, int]:
    """Map normalized canonical_name -> artist_id for the CF vocab.

    Multiple canonicals can normalize to the same key; we keep the lowest
    artist_id (which corresponds to the most-played artist in the source data).
    """
    if lookup is None:
        lookup = pd.read_parquet(ARTIST_LOOKUP_PATH)
    df = lookup.assign(norm=lookup["canonical_name"].map(normalize_artist))
    df = df.sort_values("artist_id").drop_duplicates("norm", keep="first")
    return dict(zip(df["norm"], df["artist_id"].astype(int)))


def _route_one_range(
    artists: Iterable[dict | str],
    *,
    time_range_weight: float,
    norm_to_id: dict[str, int],
    norm_keys: list[str],
    cache: ArtistCache,
    proxy_decay: float,
    fuzzy_threshold: int,
) -> tuple[dict[int, float], dict[int, float], list[dict], list[str]]:
    direct: dict[int, float] = {}
    proxies: dict[int, float] = {}
    routing: list[dict] = []
    unmatched: list[str] = []

    for sa in artists:
        name = sa["name"] if isinstance(sa, dict) else str(sa)
        norm = normalize_artist(name)

        # 1. Exact normalized match
        if norm in norm_to_id:
            aid = norm_to_id[norm]
            direct[aid] = direct.get(aid, 0.0) + time_range_weight
            routing.append({"spotify": name, "match": "exact", "cf_aid": aid})
            continue

        # 2. Fuzzy match
        hit = process.extractOne(norm, norm_keys, scorer=fuzz.ratio, score_cutoff=fuzzy_threshold)
        if hit:
            matched_norm, score, _ = hit
            aid = norm_to_id[matched_norm]
            direct[aid] = direct.get(aid, 0.0) + time_range_weight
            routing.append({
                "spotify": name, "match": f"fuzzy({score:.0f})",
                "cf_aid": aid, "matched_to": matched_norm,
            })
            continue

        # 3. Proxy substitution
        try:
            similars = cache.get_lastfm_similar(name, limit=50)
        except Exception as e:
            routing.append({"spotify": name, "match": "error", "error": str(e)})
            unmatched.append(name)
            continue

        added = 0
        for _, canonical, score in similars:
            pn = normalize_artist(canonical)
            if pn in norm_to_id:
                aid = norm_to_id[pn]
                w = time_range_weight * proxy_decay * float(score)
                proxies[aid] = proxies.get(aid, 0.0) + w
                added += 1

        if added > 0:
            routing.append({"spotify": name, "match": f"proxy({added})"})
        else:
            routing.append({"spotify": name, "match": "none"})
            unmatched.append(name)

    return direct, proxies, routing, unmatched


def route_artists(
    short_term: Iterable[dict | str],
    medium_term: Iterable[dict | str],
    *,
    cache: ArtistCache,
    norm_to_id: dict[str, int] | None = None,
    proxy_decay: float = PROXY_DECAY_DEFAULT,
    fuzzy_threshold: int = FUZZY_THRESHOLD_DEFAULT,
    short_weight: float = SHORT_TERM_WEIGHT,
    medium_weight: float = MEDIUM_TERM_WEIGHT,
) -> RoutingResult:
    """Route Spotify top artists from both time ranges into CF weights."""
    if norm_to_id is None:
        norm_to_id = build_norm_to_id()
    norm_keys = list(norm_to_id.keys())

    s_direct, s_proxies, s_routing, s_unmatched = _route_one_range(
        short_term, time_range_weight=short_weight, norm_to_id=norm_to_id,
        norm_keys=norm_keys, cache=cache, proxy_decay=proxy_decay,
        fuzzy_threshold=fuzzy_threshold,
    )
    m_direct, m_proxies, m_routing, m_unmatched = _route_one_range(
        medium_term, time_range_weight=medium_weight, norm_to_id=norm_to_id,
        norm_keys=norm_keys, cache=cache, proxy_decay=proxy_decay,
        fuzzy_threshold=fuzzy_threshold,
    )

    for r in s_routing: r["range"] = "short"
    for r in m_routing: r["range"] = "medium"

    direct: dict[int, float] = {}
    for aid, w in {**s_direct}.items(): direct[aid] = direct.get(aid, 0) + w
    for aid, w in m_direct.items():     direct[aid] = direct.get(aid, 0) + w
    proxies: dict[int, float] = {}
    for aid, w in s_proxies.items():    proxies[aid] = proxies.get(aid, 0) + w
    for aid, w in m_proxies.items():    proxies[aid] = proxies.get(aid, 0) + w

    weights = dict(direct)
    for aid, w in proxies.items():
        weights[aid] = weights.get(aid, 0.0) + w

    return RoutingResult(
        weights=weights,
        routing=s_routing + m_routing,
        unmatched=s_unmatched + m_unmatched,
        n_direct=len(direct),
        n_proxy_only=len(set(proxies) - set(direct)),
    )


def expand_to_modern(
    cf_recs: pd.DataFrame,
    *,
    cache: ArtistCache,
    norm_to_id: dict[str, int],
    exclude_norms: set[str] | None = None,
    n: int = 20,
    similar_limit: int = 50,
) -> pd.DataFrame:
    """Reverse-proxy: surface artists NOT in CF vocab (mostly post-2009) by
    walking Last.fm similars from the top CF recs.

    For each (cf_canonical, cf_score) in `cf_recs`, fetch its Last.fm similar
    list. For each similar artist that is NOT in CF vocab and NOT in the user's
    own input, accumulate `cf_score * lastfm_similarity`. Sort, return top-N.

    `exclude_norms`: normalized names of the user's Spotify inputs — drop these
    from the output so we don't recommend artists they already listen to.
    """
    if exclude_norms is None:
        exclude_norms = set()

    accumulated: dict[str, float] = {}
    support_count: dict[str, int] = {}
    support_seeds: dict[str, list[tuple[str, float]]] = {}
    display_name: dict[str, str] = {}

    for row in cf_recs.itertuples(index=False):
        seed_name = row.canonical_name
        seed_score = float(row.score)
        try:
            similars = cache.get_lastfm_similar(seed_name, limit=similar_limit)
        except Exception:
            continue

        for _, canonical, sim_score in similars:
            norm = normalize_artist(canonical)
            if not norm or norm in norm_to_id or norm in exclude_norms:
                continue
            if is_denylisted(canonical):
                continue
            contrib = seed_score * float(sim_score)
            accumulated[norm] = accumulated.get(norm, 0.0) + contrib
            support_count[norm] = support_count.get(norm, 0) + 1
            support_seeds.setdefault(norm, []).append((seed_name, float(sim_score)))
            display_name.setdefault(norm, canonical)

    if not accumulated:
        return pd.DataFrame(
            columns=["name", "score", "supporting_cf_recs", "support_seeds"]
        )

    df = pd.DataFrame(
        [
            (display_name[k], v, support_count[k], support_seeds[k])
            for k, v in accumulated.items()
        ],
        columns=["name", "score", "supporting_cf_recs", "support_seeds"],
    )
    df = df.sort_values("score", ascending=False).head(n).reset_index(drop=True)
    return df
