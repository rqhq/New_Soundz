"""Align my artist plays with the Last.fm 360K user-artist matrix.

Pipeline:
  1. Convert the raw 1.5GB TSV (`usersha1-artmbid-artname-plays.tsv`) to Parquet on
     first run; delete the TSV afterwards (configurable).
  2. Build a canonical artist universe by grouping Last.fm rows on a normalized
     artist name (lowercase, no punctuation, no leading "the "). Within each
     group, pick the most common original spelling as the canonical name.
  3. Match my listened-to artists into that universe — exact match on the
     normalized name first, then rapidfuzz for the leftovers.
  4. Emit `interactions.parquet` (`user_id, artist_id, count`) with all Last.fm
     users plus `me`, and `artist_lookup.parquet` mapping artist_id ↔ name.

This is the input the LensKit pipeline consumes on Day 3.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pa_parquet
from rapidfuzz import fuzz, process

LASTFM_TSV_DEFAULT = Path("data/raw/lastfm/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv")
LASTFM_PARQUET_DEFAULT = Path("data/raw/lastfm/lastfm_360k.parquet")

# rapidfuzz score (0-100) below which we treat the match as garbage.
# Using fuzz.ratio (length-sensitive Levenshtein) — token_set_ratio over-matches
# substring overlaps (e.g., "malcolm todd" → "todd" scores 100, which is wrong).
# Threshold is conservative because false positives mis-attribute my plays to
# the wrong artist (e.g., "George Clanton" → "George Clinton" at score 92 is
# a 248-play vaporwave→P-Funk misrouting). Better to drop a real edge case
# than pollute the user's taste vector.
FUZZY_THRESHOLD = 95

_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_artist(name: str | None) -> str:
    """Loose canonicalization for matching across data sources.

    Lowercase, NFKD-normalized, punctuation-stripped, leading 'the ' dropped,
    whitespace collapsed. Designed to collapse "JAY-Z"/"Jay Z"/"jay-z" and
    "The Beatles"/"Beatles, The" into a single key.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = s.encode("ascii", "ignore").decode("ascii")  # drop accents
    s = s.lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    if s.startswith("the "):
        s = s[4:]
    if s.endswith(", the"):
        s = s[:-5]
    return s


def convert_lastfm_tsv_to_parquet(
    tsv_path: Path = LASTFM_TSV_DEFAULT,
    parquet_path: Path = LASTFM_PARQUET_DEFAULT,
    drop_tsv: bool = True,
) -> Path:
    """Read the raw Last.fm 360K TSV and write it as Parquet.

    Skips work if the parquet already exists. If drop_tsv=True, removes the TSV
    after a successful write — Parquet is ~5-10x smaller and faster to load.
    """
    if parquet_path.exists():
        print(f"  parquet already exists: {parquet_path} — skipping conversion")
        return parquet_path

    if not tsv_path.exists():
        raise FileNotFoundError(f"Last.fm TSV not found: {tsv_path}")

    print(f"  reading {tsv_path} ...")
    table = pa_csv.read_csv(
        tsv_path,
        read_options=pa_csv.ReadOptions(
            column_names=["user_sha1", "artist_mbid", "artist_name", "plays"],
            block_size=1 << 24,  # 16MB blocks
        ),
        parse_options=pa_csv.ParseOptions(delimiter="\t"),
        convert_options=pa_csv.ConvertOptions(
            column_types={
                "user_sha1": pa.string(),
                "artist_mbid": pa.string(),
                "artist_name": pa.string(),
                "plays": pa.int32(),
            },
            null_values=["", "\\N"],
            strings_can_be_null=True,
        ),
    )
    print(f"  parsed {table.num_rows:,} rows")

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pa_parquet.write_table(table, parquet_path, compression="snappy")
    print(f"  wrote {parquet_path} ({parquet_path.stat().st_size / 1e6:.0f} MB)")

    if drop_tsv:
        tsv_path.unlink()
        print(f"  dropped {tsv_path}")

    return parquet_path


def build_artist_universe(lastfm: pd.DataFrame) -> pd.DataFrame:
    """Collapse Last.fm artists by normalized name.

    Returns one row per canonical artist with:
      artist_id          int       — dense index assigned here
      norm_name          str       — normalized matching key
      canonical_name     str       — most common original spelling
      total_plays        int       — sum across all users (for tie-breaking)
      n_users            int       — distinct users that played them
    """
    df = lastfm.copy()
    df["norm_name"] = df["artist_name"].map(normalize_artist)
    df = df[df["norm_name"] != ""]  # drop blank-after-normalization rows

    # canonical_name = most-frequent (artist_name, plays-weighted) per norm group
    spelling_counts = (
        df.groupby(["norm_name", "artist_name"], as_index=False)["plays"]
        .sum()
        .sort_values(["norm_name", "plays"], ascending=[True, False])
    )
    canonical = spelling_counts.drop_duplicates("norm_name", keep="first")[
        ["norm_name", "artist_name"]
    ].rename(columns={"artist_name": "canonical_name"})

    agg = df.groupby("norm_name", as_index=False).agg(
        total_plays=("plays", "sum"),
        n_users=("user_sha1", "nunique"),
    )
    universe = agg.merge(canonical, on="norm_name")
    universe = universe.sort_values("total_plays", ascending=False).reset_index(drop=True)
    universe["artist_id"] = universe.index.astype("int32")
    return universe[["artist_id", "norm_name", "canonical_name", "total_plays", "n_users"]]


def match_my_artists(
    my_artists: pd.DataFrame,
    universe: pd.DataFrame,
    fuzzy_threshold: int = FUZZY_THRESHOLD,
) -> pd.DataFrame:
    """Map my artist names → Last.fm artist_ids.

    Two-stage:
      1. Exact match on the normalized name (catches the vast majority).
      2. rapidfuzz on the remainder; accept if score >= threshold.

    Returns one row per input artist with:
      artist_name (mine)  — original spelling
      norm_name           — normalized form
      artist_id           — assigned id from universe, or <NA>
      method              — 'exact' | 'fuzzy' | 'unmatched'
      match_score         — 100 for exact, fuzz score for fuzzy, NA for unmatched
      matched_canonical   — Last.fm canonical name we mapped to (or NA)
    """
    my = my_artists.copy()
    my["norm_name"] = my["artist_name"].map(normalize_artist)

    norm_to_id = dict(zip(universe["norm_name"], universe["artist_id"]))
    norm_to_canonical = dict(zip(universe["norm_name"], universe["canonical_name"]))

    methods, ids, scores, canonicals = [], [], [], []
    universe_norms = universe["norm_name"].tolist()  # rapidfuzz needs a list

    for nm in my["norm_name"]:
        if nm == "":
            methods.append("unmatched"); ids.append(pd.NA); scores.append(pd.NA); canonicals.append(pd.NA)
            continue
        if nm in norm_to_id:
            methods.append("exact"); ids.append(norm_to_id[nm]); scores.append(100); canonicals.append(norm_to_canonical[nm])
            continue
        # rapidfuzz against the universe with fuzz.ratio — penalizes length
        # differences so substring matches don't get full marks.
        best = process.extractOne(nm, universe_norms, scorer=fuzz.ratio, score_cutoff=fuzzy_threshold)
        if best is None:
            methods.append("unmatched"); ids.append(pd.NA); scores.append(pd.NA); canonicals.append(pd.NA)
        else:
            matched_norm, score, _idx = best
            methods.append("fuzzy")
            ids.append(norm_to_id[matched_norm])
            scores.append(int(score))
            canonicals.append(norm_to_canonical[matched_norm])

    my["artist_id"] = pd.array(ids, dtype="Int32")
    my["method"] = methods
    my["match_score"] = pd.array(scores, dtype="Int16")
    my["matched_canonical"] = canonicals
    return my


def build_interactions(
    lastfm: pd.DataFrame,
    my_artists_matched: pd.DataFrame,
    universe: pd.DataFrame,
    me_user_id: str = "me",
) -> pd.DataFrame:
    """Produce the unified (user_id, artist_id, count) interaction frame.

    All Last.fm rows + my matched plays. Last.fm rows are aggregated by
    (user_sha1, normalized_artist_name) since multiple original spellings
    can collapse to one canonical artist.
    """
    norm_to_id = dict(zip(universe["norm_name"], universe["artist_id"]))

    lf = lastfm.copy()
    lf["norm_name"] = lf["artist_name"].map(normalize_artist)
    lf = lf[lf["norm_name"] != ""]
    lf["artist_id"] = lf["norm_name"].map(norm_to_id).astype("Int32")
    lf = (
        lf.groupby(["user_sha1", "artist_id"], as_index=False)["plays"]
        .sum()
        .rename(columns={"user_sha1": "user_id", "plays": "count"})
    )

    me = my_artists_matched.dropna(subset=["artist_id"]).copy()
    # When my artist set normalizes onto the same Last.fm canonical (rare but
    # possible — e.g., "JAY-Z" and "Jay Z" both in my history), aggregate plays.
    me = me.groupby("artist_id", as_index=False)["play_count"].sum()
    me = me.rename(columns={"play_count": "count"})
    me["user_id"] = me_user_id
    me = me[["user_id", "artist_id", "count"]]

    interactions = pd.concat([lf, me], ignore_index=True)
    interactions["count"] = interactions["count"].astype("int32")
    return interactions


def run(
    my_plays_path: Path = Path("data/processed/plays_by_artist.parquet"),
    lastfm_tsv: Path = LASTFM_TSV_DEFAULT,
    lastfm_parquet: Path = LASTFM_PARQUET_DEFAULT,
    out_dir: Path = Path("data/processed"),
    drop_tsv: bool = True,
) -> dict[str, Path]:
    """End-to-end Day 2 pipeline. Returns {name: path} of written files."""
    print("Stage 1: ensure Last.fm TSV is in Parquet ...")
    convert_lastfm_tsv_to_parquet(lastfm_tsv, lastfm_parquet, drop_tsv=drop_tsv)

    print("\nStage 2: load datasets")
    lastfm = pd.read_parquet(lastfm_parquet)
    my_artists = pd.read_parquet(my_plays_path)
    print(f"  Last.fm: {len(lastfm):,} rows, {lastfm['user_sha1'].nunique():,} users, "
          f"{lastfm['artist_name'].nunique():,} unique artist strings")
    print(f"  Mine:    {len(my_artists):,} artists, {my_artists['play_count'].sum():,} plays")

    print("\nStage 3: build artist universe")
    universe = build_artist_universe(lastfm)
    print(f"  collapsed to {len(universe):,} canonical artists")

    print("\nStage 4: match my artists into the universe")
    matched = match_my_artists(my_artists, universe)
    method_counts = matched["method"].value_counts()
    print(f"  exact:     {method_counts.get('exact', 0):,}")
    print(f"  fuzzy:     {method_counts.get('fuzzy', 0):,}")
    print(f"  unmatched: {method_counts.get('unmatched', 0):,}")

    matched_plays = matched.dropna(subset=["artist_id"])["play_count"].sum()
    total_plays = matched["play_count"].sum()
    print(f"  play-weighted coverage: {matched_plays:,}/{total_plays:,} ({matched_plays/total_plays:.1%})")

    print("\nStage 5: build interactions matrix")
    interactions = build_interactions(lastfm, matched, universe)
    artist_lookup = universe[["artist_id", "canonical_name"]].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "interactions": out_dir / "interactions.parquet",
        "artist_lookup": out_dir / "artist_lookup.parquet",
        "my_match_audit": out_dir / "my_artist_matches.parquet",
    }
    interactions.to_parquet(paths["interactions"], index=False)
    artist_lookup.to_parquet(paths["artist_lookup"], index=False)
    matched.to_parquet(paths["my_match_audit"], index=False)
    for name, p in paths.items():
        print(f"  wrote {p} ({p.stat().st_size / 1e6:.1f} MB)")

    print("\nStage 6: sanity check the matrix")
    n_users = interactions["user_id"].nunique()
    n_artists = interactions["artist_id"].nunique()
    nnz = len(interactions)
    density = nnz / (n_users * n_artists)
    print(f"  users:    {n_users:,}")
    print(f"  artists:  {n_artists:,}")
    print(f"  nnz:      {nnz:,}")
    print(f"  density:  {density:.6f}  ({density*100:.4f}%)")

    return paths


if __name__ == "__main__":
    run()
