"""Streamlit entrypoint — Step 1: OAuth hello-world.

Run from repo root with:

    uv run streamlit run app/main.py --server.port=5000

The Spotify dashboard's redirect URI must match the URL Streamlit serves on
exactly. We use port 5000 to reuse the existing http://127.0.0.1:5000/callback
registration. Streamlit serves any path under its root, so /callback works.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure src/ is importable when run via `streamlit run app/main.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import spotipy
import streamlit as st
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth

load_dotenv(REPO_ROOT / ".env")

CLIENT_ID = os.environ["SPOTIFY_CLIENT_ID"]
CLIENT_SECRET = os.environ["SPOTIFY_CLIENT_SECRET"]
REDIRECT_URI = os.environ["SPOTIFY_REDIRECT_URI"]
SCOPES = "user-top-read user-read-recently-played user-read-private user-read-currently-playing"

# Streamlit sessions can be lost when the browser navigates to Spotify and back,
# so we use a file-based token cache instead of MemoryCacheHandler. Local dev
# only — fine for a single-user laptop setup. .gitignored.
TOKEN_CACHE_PATH = REPO_ROOT / ".spotify_token_cache.json"

st.set_page_config(page_title="Spotify Recs", layout="wide")


def _build_auth_manager() -> SpotifyOAuth:
    return SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        cache_handler=spotipy.cache_handler.CacheFileHandler(
            cache_path=str(TOKEN_CACHE_PATH)
        ),
        open_browser=False,
    )


def _get_authenticated_client() -> spotipy.Spotify | None:
    """Run the full OAuth flow inline. Returns a client once authenticated."""
    existing = st.session_state.get("auth_manager")
    if not isinstance(existing, SpotifyOAuth):
        st.session_state.auth_manager = _build_auth_manager()
    auth = st.session_state.auth_manager

    # Already authenticated this session, OR there's a valid cached token from a
    # prior session (file-based cache survives browser/Streamlit restarts).
    if st.session_state.get("authenticated") or auth.validate_token(
        auth.cache_handler.get_cached_token()
    ):
        st.session_state.authenticated = True
        return spotipy.Spotify(auth_manager=auth)

    # Did Spotify just redirect back here?
    err = st.query_params.get("error")
    code = st.query_params.get("code")
    if err:
        st.query_params.clear()
        st.warning(f"Spotify returned an error: {err}. Try clicking Connect again.")
    elif code:
        try:
            auth.get_access_token(code, check_cache=False)
            st.session_state.authenticated = True
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.query_params.clear()
            st.warning(f"Login expired, please try again. (details: {e})")

    # Fresh visit — show the login link.
    auth_url = auth.get_authorize_url()
    st.title("Spotify Recommender")
    st.write("Personal artist recommendations from your Spotify listening history.")
    st.markdown(f"[**Connect Spotify →**]({auth_url})")
    st.caption(
        "This app is in Spotify dev mode (max 25 allowlisted users). "
        "If you can't log in, you're not on the allowlist yet."
    )
    return None


def main() -> None:
    sp = _get_authenticated_client()
    if sp is None:
        return

    me = sp.current_user()
    st.success(f"Logged in as **{me['display_name']}** (id={me['id']})")

    st.subheader("Top 10 artists (last 4 weeks)")
    top = sp.current_user_top_artists(limit=10, time_range="short_term")["items"]
    for i, a in enumerate(top, 1):
        cols = st.columns([1, 4])
        with cols[0]:
            if a.get("images"):
                st.image(a["images"][-1]["url"], width=60)
        with cols[1]:
            genres = ", ".join(a["genres"][:3]) if a["genres"] else "(no genres listed)"
            st.write(f"**{i}. {a['name']}** — {genres}")

    if st.button("Log out"):
        for key in ("auth_manager", "authenticated"):
            st.session_state.pop(key, None)
        TOKEN_CACHE_PATH.unlink(missing_ok=True)
        st.rerun()


if __name__ == "__main__":
    main()
