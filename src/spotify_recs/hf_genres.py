"""HuggingFace Spotify Tracks dataset → artist→genres lookup.

The dataset (`maharshipandya/spotify-tracks-dataset`) contains ~114k tracks
each tagged with one of 125 playlist-derived genres. We collapse it to
`{normalized_artist: set[genre]}` and persist as parquet so the cache layer
can do constant-time lookups without re-parsing the CSV every cold start.

Usage:
    uv run python -m spotify_recs.hf_genres   # downloads + builds lookup
    HFGenreLookup().get("tyler the creator")  # → {'hip-hop'}

The 125 genres are also exposed as `HF_GENRE_TAXONOMY` so other modules can
use them as the allowlist for filtering noisy Last.fm tags.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from spotify_recs.align import normalize_artist

DATASET_URL = (
    "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset"
    "/resolve/main/dataset.csv"
)
RAW_PATH = Path("data/raw/hf_spotify_tracks.csv")
LOOKUP_PATH = Path("data/processed/hf_artist_genres.parquet")


def download_dataset(force: bool = False) -> Path:
    if RAW_PATH.exists() and not force:
        return RAW_PATH
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_URL} ...")
    r = requests.get(DATASET_URL, timeout=60)
    r.raise_for_status()
    RAW_PATH.write_bytes(r.content)
    print(f"  saved {RAW_PATH} ({RAW_PATH.stat().st_size / 1e6:.1f} MB)")
    return RAW_PATH


def build_lookup(force: bool = False) -> pd.DataFrame:
    """Group HF dataset by normalized artist, collect genre frequencies.

    Returns a DataFrame with columns [artist_key, canonical_name, genres]
    where genres is a list[(genre, count)] sorted by count desc. Counts let
    downstream scoring downweight noise — the dataset is playlist-derived
    so a single off-genre track can pollute an artist's tags.
    """
    if LOOKUP_PATH.exists() and not force:
        return pd.read_parquet(LOOKUP_PATH)

    download_dataset()
    df = pd.read_csv(RAW_PATH)
    # The dataset uses ';' to separate collaborators in `artists`. We expand
    # so each artist gets credit for every track they appear on.
    df = df[["artists", "track_genre"]].dropna()
    df = df.assign(artist=df["artists"].str.split(";")).explode("artist")
    df["artist"] = df["artist"].str.strip()
    df = df[df["artist"] != ""]
    df["artist_key"] = df["artist"].map(normalize_artist)
    df = df[df["artist_key"] != ""]

    def _genre_counts(s: pd.Series) -> list[tuple[str, int]]:
        vc = s.value_counts()
        return [(g, int(c)) for g, c in vc.items()]

    grouped = (
        df.groupby("artist_key")
        .agg(
            canonical_name=("artist", lambda s: s.value_counts().idxmax()),
            genres=("track_genre", _genre_counts),
        )
        .reset_index()
    )
    # Parquet can't store list[tuple] — serialize the genres column to JSON.
    grouped["genres"] = grouped["genres"].map(json.dumps)

    LOOKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_parquet(LOOKUP_PATH, index=False)
    avg_genres = grouped["genres"].map(lambda s: len(json.loads(s))).mean()
    print(
        f"  built lookup: {len(grouped):,} artists, "
        f"{avg_genres:.1f} avg genres/artist, "
        f"saved {LOOKUP_PATH}"
    )
    return grouped


def build_taxonomy() -> set[str]:
    """Distinct genres in the HF dataset.

    Used as the allowlist when filtering noisy Last.fm tags down to real
    genre labels. Computed from the lookup so it stays in sync if the
    dataset is ever rebuilt.
    """
    df = build_lookup()
    all_genres: set[str] = set()
    for raw in df["genres"]:
        for g, _ in json.loads(raw):
            all_genres.add(g)
    return all_genres


class HFGenreLookup:
    """Read-only artist→[(genre, count), ...] lookup."""

    def __init__(self, df: pd.DataFrame | None = None):
        if df is None:
            df = build_lookup()
        self._index: dict[str, list[tuple[str, int]]] = {
            k: [tuple(t) for t in json.loads(raw)]
            for k, raw in zip(df["artist_key"], df["genres"])
        }
        self._taxonomy: set[str] | None = None

    def get(self, artist: str) -> list[tuple[str, int]]:
        return list(self._index.get(normalize_artist(artist), []))

    @property
    def taxonomy(self) -> set[str]:
        if self._taxonomy is None:
            tax: set[str] = set()
            for gs in self._index.values():
                for g, _ in gs:
                    tax.add(g)
            self._taxonomy = tax
        return self._taxonomy


HF_GENRE_TAXONOMY: set[str] | None = None


def get_taxonomy() -> set[str]:
    """Module-level lazy taxonomy access."""
    global HF_GENRE_TAXONOMY
    if HF_GENRE_TAXONOMY is None:
        HF_GENRE_TAXONOMY = build_taxonomy()
    return HF_GENRE_TAXONOMY


if __name__ == "__main__":
    df = build_lookup(force=False)
    tax = get_taxonomy()
    print(f"\nTaxonomy: {len(tax)} distinct genres")
    print(f"Sample: {sorted(tax)[:15]}")

    print("\n=== Spot checks ===")
    lookup = HFGenreLookup(df)
    for name in [
        "Tyler, The Creator", "Frank Ocean", "Kendrick Lamar",
        "Daft Punk", "Radiohead", "Kanye West",
        "Malcolm Todd", "HYUKOH", "JPEGMAFIA",
    ]:
        print(f"  {name:25s} -> {lookup.get(name)}")
