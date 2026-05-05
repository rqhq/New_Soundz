"""Spotify OAuth helper.

Uses Spotipy's auth code flow with PKCE. Reads credentials from .env at repo
root. Caches tokens in .spotify_token_cache.json (gitignored). Run as a script
to verify the flow end-to-end:

    uv run python -m spotify_recs.spotify_auth
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyPKCE

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKEN_CACHE_PATH = REPO_ROOT / ".spotify_token_cache.json"

SCOPES = [
    "user-top-read",
    "user-read-recently-played",
    "user-read-private",
]


def get_spotify_client() -> spotipy.Spotify:
    load_dotenv(REPO_ROOT / ".env")

    client_id = os.environ["SPOTIFY_CLIENT_ID"]
    redirect_uri = os.environ["SPOTIFY_REDIRECT_URI"]

    auth_manager = SpotifyPKCE(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=" ".join(SCOPES),
        cache_handler=spotipy.cache_handler.CacheFileHandler(
            cache_path=str(TOKEN_CACHE_PATH)
        ),
        open_browser=True,
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def _hello_world() -> int:
    sp = get_spotify_client()

    me = sp.current_user()
    print(f"Logged in as: {me['display_name']} (id={me['id']})")
    print()

    top = sp.current_user_top_artists(limit=10, time_range="short_term")
    print("Top 10 artists (last 4 weeks):")
    for i, a in enumerate(top["items"], 1):
        genres = ", ".join(a["genres"][:3]) if a["genres"] else "(no genres)"
        print(f"  {i:2d}. {a['name']:30s}  [{genres}]")

    return 0


if __name__ == "__main__":
    sys.exit(_hello_world())
