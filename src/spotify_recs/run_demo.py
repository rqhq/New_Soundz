"""End-to-end CLI demo: log in -> fetch tops -> route -> fold-in -> print recs.

    uv run python -m spotify_recs.run_demo

No UI yet. This is the validation rig for the live recommender.
"""

from __future__ import annotations

import sys

from spotify_recs.align import normalize_artist
from spotify_recs.cache import ArtistCache
from spotify_recs.recommender import load_pipeline, recommend_for_history
from spotify_recs.routing import build_norm_to_id, expand_to_modern, route_artists
from spotify_recs.spotify_auth import get_spotify_client

TOP_LIMIT = 50
N_RECS = 20


def main() -> int:
    print("Loading model and CF vocab...")
    pipe = load_pipeline()
    norm_to_id = build_norm_to_id()
    print(f"  CF vocab: {len(norm_to_id)} unique normalized artist names")

    print("Authenticating with Spotify...")
    sp = get_spotify_client()
    me = sp.current_user()
    print(f"  Logged in as: {me['display_name']}")

    print(f"Fetching top {TOP_LIMIT} artists (short + medium term)...")
    short = sp.current_user_top_artists(limit=TOP_LIMIT, time_range="short_term")["items"]
    medium = sp.current_user_top_artists(limit=TOP_LIMIT, time_range="medium_term")["items"]
    print(f"  short_term: {len(short)}    medium_term: {len(medium)}")

    print("Routing through match-or-proxy...")
    with ArtistCache() as cache:
        result = route_artists(short, medium, cache=cache, norm_to_id=norm_to_id)

    n_short_direct = sum(1 for r in result.routing if r["range"] == "short" and r["match"].startswith(("exact", "fuzzy")))
    n_short_proxy = sum(1 for r in result.routing if r["range"] == "short" and r["match"].startswith("proxy"))
    n_short_none = sum(1 for r in result.routing if r["range"] == "short" and r["match"] in ("none", "error"))
    n_med_direct = sum(1 for r in result.routing if r["range"] == "medium" and r["match"].startswith(("exact", "fuzzy")))
    n_med_proxy = sum(1 for r in result.routing if r["range"] == "medium" and r["match"].startswith("proxy"))
    n_med_none = sum(1 for r in result.routing if r["range"] == "medium" and r["match"] in ("none", "error"))

    print(f"  short_term:  direct={n_short_direct}  proxy={n_short_proxy}  unmatched={n_short_none}")
    print(f"  medium_term: direct={n_med_direct}  proxy={n_med_proxy}  unmatched={n_med_none}")
    print(f"  -> {len(result.weights)} unique CF artists in user vector "
          f"({result.n_direct} direct + {result.n_proxy_only} proxy-only)")
    if result.unmatched:
        preview = result.unmatched[:8]
        more = "" if len(result.unmatched) <= 8 else f" (+{len(result.unmatched)-8} more)"
        print(f"  unmatched: {preview}{more}")

    print()
    print("Top routing decisions (first 12):")
    for r in result.routing[:12]:
        rng = r["range"][0]
        info = r.get("matched_to", "") or r.get("error", "")
        info = f" -> {info}" if info else ""
        print(f"  [{rng}] {r['spotify']:30s}  {r['match']}{info}")

    if len(result.weights) < 5:
        print("\nUser vector is too sparse for CF (<5 artists). Content fallback would kick in here.")
        print("Day 4 step 4 (genre enrichment + content fallback) is not implemented yet.")
        return 1

    print("\nRunning fold-in inference...")
    recs = recommend_for_history(pipe, result.weights, n=N_RECS)

    print(f"\nClassic picks (top {N_RECS}, from CF — bounded by 2009 vocab):")
    for i, row in enumerate(recs.itertuples(index=False), 1):
        print(f"  {i:2d}. {row.canonical_name:30s}  score={row.score:.3f}")

    print("\nExpanding to modern picks via reverse-proxy (this can hit Last.fm)...")
    user_input_norms = {
        normalize_artist(sa["name"]) for sa in (short + medium)
    }
    expanded_recs = recommend_for_history(pipe, result.weights, n=50)
    with ArtistCache() as cache:
        modern = expand_to_modern(
            expanded_recs,
            cache=cache,
            norm_to_id=norm_to_id,
            exclude_norms=user_input_norms,
            n=N_RECS,
        )

    if modern.empty:
        print("  (no modern candidates surfaced — Last.fm similars all already in CF vocab)")
    else:
        print(f"\nModern picks (top {N_RECS}, post-2009 candidates via Last.fm similars):")
        for i, row in enumerate(modern.itertuples(index=False), 1):
            print(f"  {i:2d}. {row.name:30s}  score={row.score:.3f}  "
                  f"(supported by {row.supporting_cf_recs} CF recs)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
