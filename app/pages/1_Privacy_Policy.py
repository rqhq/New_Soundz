"""Privacy policy page. Required by the Spotify Developer Dashboard review."""

import streamlit as st

st.set_page_config(page_title="Privacy Policy · New Soundz", layout="wide")

st.title("Privacy Policy")
st.caption("Last updated: 2026-05-07")

st.markdown(
    """
**New Soundz** is a personal portfolio project that surfaces music
recommendations based on your Spotify listening history. This policy explains
what data the app sees, what it does with it, and what it doesn't do.

## What we collect

When you sign in via Spotify OAuth, the app requests these scopes:

- `user-top-read` — your top artists and tracks across short / medium / long term windows
- `user-read-recently-played` — your last 50 plays
- `user-read-currently-playing` — what's playing right now
- `user-read-private` — your display name, profile image, follower count, and account ID

The app does **not** request access to your email, payment information,
playlists, library, friends list, or any other Spotify data outside the
scopes listed above.

## What we do with it

- **In demo mode (the default)**, all visitors view the developer's (rqhq's)
  data — your own Spotify account is never touched unless you explicitly
  click "Connect your own Spotify."
- **In personal mode (after you connect)**, the app fetches your top
  artists and tracks, runs them through a recommendation model, and
  renders the results in your browser session.
- All processing happens in-memory during your session. Your access
  token is held in Streamlit's session state for the duration of your
  visit and discarded when you close the tab or click "Log out."

## What we don't do

- We do **not** store your Spotify data on disk after your session ends.
- We do **not** sell, share, or transmit your data to any third party.
- We do **not** use your data for advertising or profiling.
- We do **not** train any models on your personal listening data — the
  underlying recommender is trained offline on the public Last.fm 360K
  research dataset.

## Third-party services

- **Spotify** — when you authenticate, your browser communicates directly
  with Spotify's servers. Their privacy policy applies:
  https://www.spotify.com/legal/privacy-policy/
- **Last.fm API** — the app queries Last.fm's public API for artist
  similarity data and genre tags. These queries reference artist names
  only, never your identity.
- **Streamlit Community Cloud** — the app is hosted on Streamlit's free
  tier. Their privacy policy: https://streamlit.io/privacy-policy

## Your rights and choices

- **Disconnect** — clicking "Log out" or "← Back to demo" in the sidebar
  removes the token from your session immediately.
- **Revoke access** — visit https://www.spotify.com/account/apps/ and
  remove "New Soundz" to revoke OAuth access entirely.
- **Questions** — open an issue on the project's GitHub repository.

## Scope of this app

New Soundz is a portfolio / educational project, not a commercial product.
It is currently in Spotify Developer Mode, which means access is limited
to manually allowlisted accounts (max 25). The app is provided as-is with
no guarantees of uptime or correctness.
"""
)
