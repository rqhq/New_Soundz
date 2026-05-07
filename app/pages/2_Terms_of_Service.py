"""Terms of Service page. Required by the Spotify Developer Dashboard review."""

import streamlit as st

st.set_page_config(page_title="Terms of Service · New Soundz", layout="wide")

st.title("Terms of Service")
st.caption("Last updated: 2026-05-07")

st.markdown(
    """
By using **New Soundz**, you agree to the terms below. These terms cover
both the public demo (viewing the developer's data) and personal mode
(connecting your own Spotify account).

## What this is

New Soundz is a personal portfolio / educational project demonstrating a
hybrid recommendation system built on top of the Spotify and Last.fm
public APIs. It is not a commercial product.

## Acceptable use

- You may use the app for personal, non-commercial exploration of your
  own listening data.
- You may not attempt to scrape, redistribute, or commercially exploit
  any data surfaced by the app, including the recommendations themselves.
- You may not use the app to harass, defame, or violate the rights of
  any third party.
- You may not attempt to bypass rate limits, the Spotify allowlist, or
  any other access controls.

## Spotify and Last.fm content

All artist names, track metadata, album art, profile information, and
similar content are owned by Spotify, the original rights holders, or
Last.fm's contributors. Their terms apply alongside these:

- Spotify Developer Terms: https://developer.spotify.com/terms
- Last.fm API Terms: https://www.last.fm/api/tos

## No warranty

The app is provided **as-is**, with no warranty of any kind, express or
implied. Recommendations are best-effort outputs of a statistical model
and may surface artists you dislike, find offensive, or consider
unrelated to your tastes. The model is trained on a 2008-2009 snapshot
of Last.fm listening data and is structurally limited.

The developer makes no guarantee of:
- Uptime, performance, or availability
- Accuracy or relevance of recommendations
- Compatibility with any particular browser or device
- Continued operation — the project may be taken offline at any time

## Limitation of liability

To the fullest extent permitted by law, the developer is not liable for
any damages arising from your use of the app, including but not limited
to lost data, missed musical discoveries, or reactions to surfaced
recommendations.

## Changes

These terms may be updated as the project evolves. Material changes will
be reflected in the "Last updated" date above. Continued use after a
change constitutes acceptance.

## Contact

Questions or concerns: open an issue on the project's GitHub repository.
"""
)
