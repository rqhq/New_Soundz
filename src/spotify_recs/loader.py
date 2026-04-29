"""Load Spotify streaming history from either export format.

Spotify offers two data exports:

1. **Account Data** (~30 days to receive, returns ~last year):
   Files: `StreamingHistory0.json`, `StreamingHistory1.json`, ...
   Fields: endTime, artistName, trackName, msPlayed

2. **Extended Streaming History** (~30 days to receive, full lifetime):
   Files: `Streaming_History_Audio_*.json`, `Streaming_History_Video_*.json`
   Fields: ts, master_metadata_track_name, master_metadata_album_artist_name,
           ms_played, spotify_track_uri, platform, conn_country, reason_start,
           reason_end, shuffle, skipped, offline, ...

This module detects which format is present and normalizes both to a single
internal schema so downstream code never has to care.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pandas as pd

# Spotify's own threshold for what counts as a "play" (30 seconds).
# Anything shorter is treated as a skip and dropped from the recommender input.
MIN_PLAY_MS = 30_000

# Unified output schema. Every row = one play event.
CANONICAL_COLUMNS = [
    "ts",            # pd.Timestamp, UTC
    "track_name",    # str
    "artist_name",   # str
    "album_name",    # str | None
    "ms_played",     # int
    "track_uri",     # str | None  ("spotify:track:..." — only in Extended export)
    "platform",      # str | None  (only in Extended export)
    "country",       # str | None  (only in Extended export)
    "shuffle",       # bool | None (only in Extended export)
    "skipped",       # bool | None (only in Extended export)
    "source",        # "account_data" | "extended" — which export this came from
]

ExportFormat = Literal["account_data", "extended"]


def detect_format(path: Path) -> ExportFormat:
    """Inspect a JSON file's first record to determine which export format it is."""
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not records:
        raise ValueError(f"{path} contains no records")
    first = records[0]
    # Extended uses snake_case + 'ts'; Account Data uses camelCase + 'endTime'
    if "ts" in first and "master_metadata_track_name" in first:
        return "extended"
    if "endTime" in first and "trackName" in first:
        return "account_data"
    raise ValueError(f"Could not identify export format for {path}. Keys: {list(first.keys())[:5]}")


def _load_account_data(path: Path) -> pd.DataFrame:
    """Load one StreamingHistoryN.json (Account Data format)."""
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    # endTime is naive UTC per Spotify's docs
    df["ts"] = pd.to_datetime(df["endTime"], utc=True)
    return pd.DataFrame({
        "ts": df["ts"],
        "track_name": df["trackName"].astype("string"),
        "artist_name": df["artistName"].astype("string"),
        "album_name": pd.Series([None] * len(df), dtype="string"),
        "ms_played": df["msPlayed"].astype("int64"),
        "track_uri": pd.Series([None] * len(df), dtype="string"),
        "platform": pd.Series([None] * len(df), dtype="string"),
        "country": pd.Series([None] * len(df), dtype="string"),
        "shuffle": pd.Series([None] * len(df), dtype="object"),
        "skipped": pd.Series([None] * len(df), dtype="object"),
        "source": pd.Series(["account_data"] * len(df), dtype="string"),
    })


def _load_extended(path: Path) -> pd.DataFrame:
    """Load one Streaming_History_Audio_*.json (Extended format)."""
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)

    # Skip podcasts — the recommender is music-only.
    # Podcasts have null in master_metadata_* and populated episode_* fields.
    df = df[df["master_metadata_track_name"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    return pd.DataFrame({
        "ts": pd.to_datetime(df["ts"], utc=True),
        "track_name": df["master_metadata_track_name"].astype("string"),
        "artist_name": df["master_metadata_album_artist_name"].astype("string"),
        "album_name": df["master_metadata_album_album_name"].astype("string"),
        "ms_played": df["ms_played"].astype("int64"),
        "track_uri": df["spotify_track_uri"].astype("string"),
        "platform": df["platform"].astype("string"),
        "country": df["conn_country"].astype("string"),
        "shuffle": df["shuffle"],
        "skipped": df["skipped"],
        "source": pd.Series(["extended"] * len(df), dtype="string"),
    })


def find_history_files(raw_dir: Path) -> list[Path]:
    """Find all streaming history JSON files in the raw data directory.

    Walks recursively because Spotify zips contain a nested folder structure.
    """
    candidates = []
    # Extended format
    candidates.extend(raw_dir.rglob("Streaming_History_Audio_*.json"))
    # Account Data — legacy short form (StreamingHistory0.json, ...)
    candidates.extend(raw_dir.rglob("StreamingHistory[0-9]*.json"))
    # Account Data — current form, music only (skip StreamingHistory_podcast_*.json)
    candidates.extend(raw_dir.rglob("StreamingHistory_music_*.json"))
    return sorted(candidates)


def load_streams(raw_dir: Path | str = "data/raw") -> pd.DataFrame:
    """Load and normalize all streaming history files in `raw_dir`.

    Returns a DataFrame with the canonical schema, sorted by timestamp.
    Includes ALL plays — does not filter by ms_played. Filtering is a separate
    step so we can compute analytics over skips too.
    """
    raw_dir = Path(raw_dir)
    files = find_history_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No streaming history JSON files found under {raw_dir}. "
            f"Expected files like 'StreamingHistory0.json' or "
            f"'Streaming_History_Audio_*.json'."
        )

    frames = []
    for path in files:
        fmt = detect_format(path)
        if fmt == "account_data":
            frames.append(_load_account_data(path))
        else:
            frames.append(_load_extended(path))
        print(f"  loaded {path.name} ({fmt}): {len(frames[-1]):,} rows")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("ts").reset_index(drop=True)

    # Drop rows missing artist or track (rare, but Spotify sometimes has them)
    before = len(df)
    df = df.dropna(subset=["artist_name", "track_name"]).reset_index(drop=True)
    if (dropped := before - len(df)) > 0:
        print(f"  dropped {dropped:,} rows missing artist/track")

    return df


def filter_real_plays(streams: pd.DataFrame, min_ms: int = MIN_PLAY_MS) -> pd.DataFrame:
    """Keep only plays >= `min_ms` (Spotify's threshold for 'counts as a play')."""
    return streams[streams["ms_played"] >= min_ms].reset_index(drop=True)


def aggregate_by_artist(plays: pd.DataFrame) -> pd.DataFrame:
    """Aggregate filtered plays into a per-artist interaction frame.

    This is the input format LensKit expects: one row per (user, item) pair
    with an interaction count. We only have one user, so user_id is constant
    here — we'll add the public-dataset users in day 2.
    """
    agg = (
        plays.groupby("artist_name", as_index=False)
        .agg(
            play_count=("ts", "size"),
            total_ms=("ms_played", "sum"),
            first_played=("ts", "min"),
            last_played=("ts", "max"),
        )
        .sort_values("play_count", ascending=False)
        .reset_index(drop=True)
    )
    agg["user_id"] = "me"
    agg["total_minutes"] = (agg["total_ms"] / 60_000).round(1)
    return agg[["user_id", "artist_name", "play_count", "total_minutes",
                "first_played", "last_played"]]


def aggregate_by_track(plays: pd.DataFrame) -> pd.DataFrame:
    """Per (artist, track) aggregation — handy for analytics, not for the recommender."""
    agg = (
        plays.groupby(["artist_name", "track_name"], as_index=False)
        .agg(
            play_count=("ts", "size"),
            total_ms=("ms_played", "sum"),
        )
        .sort_values("play_count", ascending=False)
        .reset_index(drop=True)
    )
    agg["total_minutes"] = (agg["total_ms"] / 60_000).round(1)
    return agg


def daily_listening(plays: pd.DataFrame, tz: str = "America/New_York") -> pd.DataFrame:
    """Daily totals — minutes listened, distinct artists, distinct tracks."""
    local = plays.copy()
    local["date"] = local["ts"].dt.tz_convert(tz).dt.date
    return (
        local.groupby("date", as_index=False)
        .agg(
            minutes=("ms_played", lambda s: round(s.sum() / 60_000, 1)),
            plays=("ts", "size"),
            unique_artists=("artist_name", "nunique"),
            unique_tracks=("track_name", "nunique"),
        )
    )


def save_processed(
    streams: pd.DataFrame,
    out_dir: Path | str = "data/processed",
) -> dict[str, Path]:
    """Run the full pipeline: filter, aggregate, save Parquet files.

    Returns a dict of {name: path} for the saved files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plays = filter_real_plays(streams)

    artifacts = {
        "all_streams": (streams, "all_streams.parquet"),
        "real_plays": (plays, "real_plays.parquet"),
        "by_artist": (aggregate_by_artist(plays), "plays_by_artist.parquet"),
        "by_track": (aggregate_by_track(plays), "plays_by_track.parquet"),
        "daily": (daily_listening(plays), "daily_listening.parquet"),
    }

    paths = {}
    for name, (df, filename) in artifacts.items():
        path = out_dir / filename
        df.to_parquet(path, index=False)
        paths[name] = path
        print(f"  wrote {filename}: {len(df):,} rows")

    return paths


if __name__ == "__main__":
    print("Loading streams...")
    streams = load_streams()
    print(f"\nTotal stream events: {len(streams):,}")
    print(f"Date range: {streams['ts'].min()} → {streams['ts'].max()}")
    print(f"\nProcessing & saving...")
    save_processed(streams)
    print("\nDone.")
