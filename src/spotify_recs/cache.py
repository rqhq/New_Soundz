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
from spotify_recs.hf_genres import HFGenreLookup
from spotify_recs.lastfm_api import LastFMClient

DEFAULT_DB_PATH = Path("data/processed/artist_cache.sqlite")

# When merging genres across sources we keep tags that appear in the HF
# allowlist OR pass a heuristic length filter. Tag normalization: lowercase,
# trim, replace '-' with ' ' so 'hip-hop' and 'hip hop' dedup.
def _normalize_tag(tag: str) -> str:
    return tag.strip().lower().replace("-", " ").replace("_", " ")

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
        hf: HFGenreLookup | None = None,
    ):
        self.conn = _connect(db_path)
        self.lastfm = lastfm or LastFMClient()
        # HF lookup is lazy — only loaded if a caller asks for HF genres.
        # Avoids paying the parquet read for cache instances that only do
        # similar-artist lookups.
        self._hf = hf
        self._hf_attempted = hf is not None

    def _hf_lookup(self) -> HFGenreLookup | None:
        if not self._hf_attempted:
            try:
                self._hf = HFGenreLookup()
            except Exception:
                self._hf = None
            self._hf_attempted = True
        return self._hf

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

    # -- HF Spotify Tracks dataset genres ----------------------------------

    def get_hf_genres(self, artist: str) -> list[tuple[str, int]]:
        """Genres from the HF Spotify Tracks dataset, cache-first.

        Returns [(genre, count), ...] sorted by count desc. Empty list if
        the artist isn't in the HF dataset.
        """
        key = normalize_artist(artist)
        row = self._row(key)
        if row and row["genres_hf"] is not None:
            return [tuple(t) for t in json.loads(row["genres_hf"])]

        hf = self._hf_lookup()
        genres = hf.get(artist) if hf is not None else []
        self._upsert(key, artist, genres_hf=json.dumps(genres))
        return genres

    # -- Spotify API genres -------------------------------------------------

    def get_spotify_genres(
        self, artist: str, sp, force: bool = False
    ) -> list[str]:
        """Genres from the Spotify Web API, cache-first.

        `sp` is a `spotipy.Spotify` client. Spotify genres are sparse and
        often empty even for huge artists, so we treat empty results as
        "fetched but missing" rather than re-querying — pass `force=True`
        to retry.
        """
        key = normalize_artist(artist)
        row = self._row(key)
        if not force and row and row["genres_spotify"] is not None:
            return list(json.loads(row["genres_spotify"]))

        try:
            results = sp.search(q=f'artist:"{artist}"', type="artist", limit=1)
            items = results.get("artists", {}).get("items", [])
            genres = [g.lower() for g in (items[0].get("genres", []) if items else [])]
        except Exception:
            genres = []
        self._upsert(
            key,
            artist,
            genres_spotify=json.dumps(genres),
            spotify_fetched_at=time.time(),
        )
        return genres

    # -- Merged multi-source genres ----------------------------------------

    def get_merged_genres(
        self,
        artist: str,
        sp=None,
        max_lastfm_tags: int = 15,
    ) -> list[str]:
        """Union of genres across HF + Last.fm + Spotify, deduped & filtered.

        Priority is implicit in *which* sources we query (cheapest first):
        HF (local lookup) → Last.fm tags (already cached for many artists
        via proxy substitution) → Spotify (only if `sp` is provided).

        Each source contributes tags; we filter Last.fm tags down to the
        ones that pass either an HF-taxonomy intersection OR a length-based
        heuristic (the existing TAG_DENYLIST already catches the obvious
        noise upstream). Final list is normalized + deduped.

        Cached as `genres_merged` — subsequent calls skip recomputation
        unless any source column changes.
        """
        key = normalize_artist(artist)
        row = self._row(key)
        if row and row["genres_merged"] is not None:
            return list(json.loads(row["genres_merged"]))

        merged: dict[str, float] = {}

        # HF entries contributed by a single track are usually playlist-derived
        # noise (e.g. Kendrick→'comedy' from one comp inclusion). Drop those
        # whenever the artist has any meaningfully-counted tag to anchor on.
        hf_entries = self.get_hf_genres(artist)
        if hf_entries:
            max_hf = max(c for _, c in hf_entries)
            min_hf = 2 if max_hf >= 3 else 1
            for genre, count in hf_entries:
                if count < min_hf:
                    continue
                ntag = _normalize_tag(genre)
                merged[ntag] = merged.get(ntag, 0.0) + float(count)

        from spotify_recs.hf_genres import get_taxonomy
        taxonomy = {_normalize_tag(g) for g in get_taxonomy()}
        # Common genre-root words. Lets us accept fine-grained Last.fm tags
        # like 'alternative hip hop' or 'cloud rap' that aren't in HF's
        # high-level 114-genre list but are obviously musical.
        genre_roots = {
            "rock", "pop", "hop", "hip hop", "rap", "soul", "jazz", "blues",
            "metal", "punk", "indie", "folk", "country", "electronic",
            "house", "techno", "ambient", "experimental", "alternative",
            "funk", "disco", "reggae", "classical", "rnb", "r&b", "trap",
            "wave", "core", "step", "gaze", "garage", "bass", "drum",
            "synth", "psych", "post", "lo fi", "lofi",
        }
        artist_norm = normalize_artist(artist)

        def _is_genrey(ntag: str) -> bool:
            if ntag in taxonomy:
                return True
            # Skip tags containing the artist's own name (e.g. "kanye west").
            if artist_norm and artist_norm in ntag:
                return False
            # Skip year tags, unicode symbols, place markers.
            if any(c.isdigit() for c in ntag):
                return False
            if any(not (c.isascii() and (c.isalnum() or c.isspace() or c in "&'-")) for c in ntag):
                return False
            tokens = set(ntag.split())
            return bool(tokens & genre_roots)

        lastfm_tags = self.get_lastfm_tags(artist)[:max_lastfm_tags]
        for tag, count in lastfm_tags:
            ntag = _normalize_tag(tag)
            if _is_genrey(ntag):
                merged[ntag] = merged.get(ntag, 0.0) + float(count) / 100.0

        if sp is not None:
            for g in self.get_spotify_genres(artist, sp):
                ntag = _normalize_tag(g)
                merged[ntag] = merged.get(ntag, 0.0) + 1.0

        ordered = [g for g, _ in sorted(merged.items(), key=lambda x: -x[1])]
        self._upsert(key, artist, genres_merged=json.dumps(ordered))
        return ordered

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
