"""Last.fm API client.

Two endpoints we actually use:
  - artist.getTopTags  → genre tags (up to 100 per artist)
  - artist.getSimilar  → similar artists with similarity scores

Rate-limited to 5 req/sec per Last.fm's published guidance. The client is
deliberately thin — caching lives in `cache.py`, retries are best-effort.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

LASTFM_API_URL = "https://ws.audioscrobbler.com/2.0/"
DEFAULT_TIMEOUT = 8  # seconds
MIN_INTERVAL_S = 0.2  # 5 req/sec

# Last.fm tag noise to drop. Crowdsourced tags are rife with non-genre labels —
# year tags, mood, "seen live", country tags, etc. Anything not matching a
# real-music-genre token gets filtered post-fetch (we'll intersect with a
# genre allowlist in the cache layer; this list catches the most obvious noise).
TAG_DENYLIST = frozenset({
    "seen live", "favorite", "favorites", "favourite", "favourites",
    "spotify", "soundcloud", "bandcamp", "youtube", "myspace",
    "male vocalists", "female vocalists", "male vocalist", "female vocalist",
    "albums i own", "owned albums", "vinyl", "cd",
    "good", "awesome", "amazing", "great", "love", "love it", "cool",
    "best", "epic", "perfect", "favourite artists", "favorite artists",
    "music", "album", "song", "songs", "artist", "artists",
    "usa", "uk", "united states", "united kingdom", "american", "british",
    "english", "japanese", "korean", "german", "french", "european", "international",
})


class LastFMError(RuntimeError):
    pass


class LastFMClient:
    """Thin Last.fm API client with built-in rate limiting."""

    def __init__(self, api_key: str | None = None, min_interval_s: float = MIN_INTERVAL_S):
        self.api_key = api_key or os.environ.get("LASTFM_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError(
                "LASTFM_API_KEY not set. Get one at https://www.last.fm/api/account "
                "and put it in .env"
            )
        self.min_interval_s = min_interval_s
        self._last_request_at = 0.0
        self._session = requests.Session()

    def _request(self, method: str, **params: Any) -> dict:
        # crude rate limiter: sleep if last request was less than min_interval ago
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)

        full = {"method": method, "api_key": self.api_key, "format": "json", **params}
        r = self._session.get(LASTFM_API_URL, params=full, timeout=DEFAULT_TIMEOUT)
        self._last_request_at = time.monotonic()

        if r.status_code != 200:
            raise LastFMError(f"{method} HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        if "error" in data:
            raise LastFMError(f"{method} API error {data.get('error')}: {data.get('message')}")
        return data

    def get_top_tags(self, artist: str, autocorrect: bool = True) -> list[tuple[str, int]]:
        """Return [(tag, count), ...] sorted by count desc. Empty list if artist unknown."""
        try:
            data = self._request("artist.gettoptags", artist=artist,
                                  autocorrect=int(autocorrect))
        except LastFMError as e:
            if "6" in str(e) or "not found" in str(e).lower():
                return []
            raise

        tags = data.get("toptags", {}).get("tag", [])
        if isinstance(tags, dict):  # singleton coerced to dict by the JSON
            tags = [tags]
        out = []
        for t in tags:
            name = t.get("name", "").strip().lower()
            if not name or name in TAG_DENYLIST:
                continue
            out.append((name, int(t.get("count", 0))))
        return out

    def get_similar(
        self, artist: str, limit: int = 100, autocorrect: bool = True
    ) -> list[tuple[str, float]]:
        """Return [(similar_artist_name, similarity_match_score 0-1), ...] sorted desc."""
        try:
            data = self._request("artist.getsimilar", artist=artist, limit=limit,
                                  autocorrect=int(autocorrect))
        except LastFMError as e:
            if "6" in str(e) or "not found" in str(e).lower():
                return []
            raise

        similar = data.get("similarartists", {}).get("artist", [])
        if isinstance(similar, dict):
            similar = [similar]
        out = []
        for s in similar:
            name = s.get("name", "").strip()
            score = float(s.get("match", 0.0))
            if name:
                out.append((name, score))
        return out
