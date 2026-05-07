"""One-off CLI: OAuth as the user (auth-code flow) and write the resulting
refresh token to .env as SPOTIFY_REFRESH_TOKEN.

Use this when:
  - You want to set/replace the demo-mode refresh token, AND
  - The streamlit token cache is contaminated (e.g. last login was a friend's
    account), AND
  - You don't want to fight Spotify's browser cookie state through the app.

Run with:

    # streamlit must NOT be running on port 8888 (or whatever your
    # SPOTIFY_REDIRECT_URI port is)
    uv run python -m spotify_recs.get_demo_refresh_token

Spotipy will open the auth URL in your default browser. Sign in as the
account you want the demo to expose. Use an incognito window or sign out
of accounts.spotify.com first if a stale account is auto-logged-in.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
TMP_CACHE = REPO_ROOT / ".spotify_demo_token_cache.json"

SCOPES = (
    "user-top-read user-read-recently-played user-read-private "
    "user-read-currently-playing"
)


def main() -> int:
    load_dotenv(ENV_PATH)
    client_id = (os.environ.get("SPOTIFY_CLIENT_ID") or "").strip()
    client_secret = (os.environ.get("SPOTIFY_CLIENT_SECRET") or "").strip()
    redirect_uri = (os.environ.get("SPOTIFY_REDIRECT_URI") or "").strip()
    if not (client_id and client_secret and redirect_uri):
        print("Missing SPOTIFY_CLIENT_ID/SECRET/REDIRECT_URI in .env", file=sys.stderr)
        return 1

    # Use a separate cache file so this doesn't clobber .spotify_token_cache.json
    # (which streamlit may be using for its personal-mode flow).
    TMP_CACHE.unlink(missing_ok=True)

    auth = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=SCOPES,
        cache_handler=spotipy.cache_handler.CacheFileHandler(
            cache_path=str(TMP_CACHE)
        ),
        open_browser=True,
    )

    print(f"Opening Spotify auth in your browser...")
    print(f"Redirect URI: {redirect_uri}")
    print("If a stale account is signed in, use an incognito window or visit")
    print("  https://accounts.spotify.com/en/logout")
    print("first, then re-run this script.")
    print()

    sp = spotipy.Spotify(auth_manager=auth)
    me = sp.current_user()
    print(f"Authenticated as: {me['display_name']}  (@{me['id']})")

    token_info = auth.cache_handler.get_cached_token()
    refresh_token = token_info.get("refresh_token") if token_info else None
    if not refresh_token:
        print("No refresh_token in token cache — auth flow may have failed.", file=sys.stderr)
        return 2

    text = ENV_PATH.read_text() if ENV_PATH.exists() else ""
    line = f"SPOTIFY_REFRESH_TOKEN={refresh_token}"
    if "SPOTIFY_REFRESH_TOKEN" in text:
        text = re.sub(r"SPOTIFY_REFRESH_TOKEN=.*", line, text)
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        text += f"\n# Demo mode — server-side refresh token\n{line}\n"
    ENV_PATH.write_text(text)
    print(f"\nWrote SPOTIFY_REFRESH_TOKEN to {ENV_PATH} (account: @{me['id']}).")
    print("Restart streamlit to pick it up.")

    TMP_CACHE.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
