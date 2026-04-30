"""SQLite cache for artist metadata, lazy-populated from external sources.

One row per artist keyed by normalized name. Each genre source lives in its own
JSON column so we can debug "which source provided this tag," and the merged
`genres_merged` column is what the recommender's content-based scorer reads.

Population is on-demand: when something asks for tags or similar artists for
an artist not yet in the cache, we hit the live API, persist the result, and
return it. Subsequent lookups are SQLite reads.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from spotify_recs.align import normalize_artist
from spotify_recs.lastfm_api import LastFMClient

DEFAULT_DB_PATH = Path("data/processed/artist_cache.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS artist_metadata (
    artist_key      TEXT PRIMARY KEY,        -- normalized name, e.g. 'tyler the creator'
    canonical_name  TEXT,                    -- best-effort original spelling
    spotify_id      TEXT,
    mbid            TEXT,
    genres_lastfm   TEXT,                    -- JSON list of (tag, count)
    genres_hf       TEXT,                    -- JSON list of strings
    genres_spotify  TEXT,                    -- JSON list of strings
    genres_wikidata TEXT,                    -- JSON list of strings
    genres_merged   TEXT,                    -- JSON list of strings (deduped union)
    similar_lastfm  TEXT,                    -- JSON list of (similar_norm, similar_canonical, score)
    lastfm_fetched_at  REAL,                 -- unix timestamp; NULL = never fetched
    similar_fetched_at REAL,
    spotify_fetched_at REAL
);
CREATE INDEX IF NOT EXISTS idx_artist_canonical ON artist_metadata(canonical_name);
"""


def _connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


class ArtistCache:
    """Lazy cache wrapping a SQLite db + a LastFMClient.

    Usage:
        cache = ArtistCache()
        tags = cache.get_lastfm_tags("Tyler, The Creator")  # API call, persisted
        tags = cache.get_lastfm_tags("Tyler, The Creator")  # SQLite read

        similar = cache.get_lastfm_similar("JPEGMAFIA")     # [(norm, canonical, score), ...]
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        lastfm: LastFMClient | None = None,
    ):
        self.conn = _connect(db_path)
        self.lastfm = lastfm or LastFMClient()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "ArtistCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- core read/upsert ---------------------------------------------------

    def _upsert(self, artist_key: str, canonical_name: str, **fields) -> None:
        """Insert-or-update. fields are the column names to set."""
        cols = ["artist_key", "canonical_name"] + list(fields.keys())
        placeholders = ", ".join("?" for _ in cols)
        update_clause = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "artist_key")
        values = [artist_key, canonical_name] + list(fields.values())
        self.conn.execute(
            f"INSERT INTO artist_metadata ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(artist_key) DO UPDATE SET {update_clause}",
            values,
        )
        self.conn.commit()

    def _row(self, artist_key: str) -> sqlite3.Row | None:
        cur = self.conn.execute(
            "SELECT * FROM artist_metadata WHERE artist_key = ?", (artist_key,)
        )
        return cur.fetchone()

    # -- Last.fm tags -------------------------------------------------------

    def get_lastfm_tags(self, artist: str) -> list[tuple[str, int]]:
        """Top tags from Last.fm for an artist, cache-first."""
        key = normalize_artist(artist)
        row = self._row(key)
        if row and row["genres_lastfm"] is not None:
            return [tuple(t) for t in json.loads(row["genres_lastfm"])]

        tags = self.lastfm.get_top_tags(artist)
        self._upsert(
            key,
            artist,
            genres_lastfm=json.dumps(tags),
            lastfm_fetched_at=time.time(),
        )
        return tags

    # -- Last.fm similar artists -------------------------------------------

    def get_lastfm_similar(
        self, artist: str, limit: int = 100
    ) -> list[tuple[str, str, float]]:
        """Similar artists from Last.fm.

        Returns [(similar_norm, similar_canonical, similarity_score), ...].
        """
        key = normalize_artist(artist)
        row = self._row(key)
        if row and row["similar_lastfm"] is not None:
            return [tuple(t) for t in json.loads(row["similar_lastfm"])]

        raw = self.lastfm.get_similar(artist, limit=limit)
        normalized = [(normalize_artist(name), name, score) for name, score in raw]
        self._upsert(
            key,
            artist,
            similar_lastfm=json.dumps(normalized),
            similar_fetched_at=time.time(),
        )
        return normalized
