"""Coarse genre bucketing.

The HF taxonomy has 114 genres, plus Last.fm contributes thousands of
fine-grained tags. For visualization we collapse to ~12 high-level buckets
that map cleanly to a categorical color palette and stay legible in a
legend.

`bucket_for_genres(genres)` walks the artist's merged genre list (already
sorted by confidence in `cache.get_merged_genres`) and returns the first
bucket that any genre maps into. This handles ambiguous cases like Frank
Ocean ['soul', 'electro', 'house'] → 'r&b/soul' (soul wins because it's
ranked first), while a hardcore electronic artist's first hit is 'electronic'.
"""

from __future__ import annotations

# Buckets and their display colors. Order roughly by prevalence so the
# legend reads naturally. Colors are picked to read on a dark background.
GENRE_BUCKETS: dict[str, str] = {
    "rap/hip-hop":   "#F59E0B",  # amber
    "r&b/soul":      "#EC4899",  # pink
    "rock":          "#EF4444",  # red
    "indie":         "#8B5CF6",  # violet
    "pop":           "#F472B6",  # pink-300
    "electronic":    "#22D3EE",  # cyan
    "metal":         "#94A3B8",  # slate
    "jazz":          "#FBBF24",  # yellow
    "classical":     "#A3E635",  # lime
    "folk/country":  "#84CC16",  # green-lime
    "latin":         "#F97316",  # orange
    "world":         "#06B6D4",  # teal
    "funk/disco":    "#FCD34D",  # gold
    "other":         "#9CA3AF",  # gray
}

# Generic tags that frequently appear as a top-1 entry on artists whose
# real bucket is something more specific. We treat these as "weak" — only
# used to assign a bucket if no primary tag matches in the top-K list.
# (e.g. JPEGMAFIA's first tag is 'industrial' but his actual bucket is
# rap/hip-hop, which we discover at position 2.)
_WEAK_TAGS: frozenset[str] = frozenset({
    "pop", "dance", "alternative", "experimental", "electronic",
    "hardcore", "industrial", "club", "happy", "chill", "party",
    "british", "british invasion",
})

# Static mapping. Keys must be the *normalized* genre form used downstream
# (lowercase, dashes/underscores → spaces). See cache._normalize_tag.
# Ordered by specificity: 'indie rock' must be checked before 'rock'.
_GENRE_TO_BUCKET: dict[str, str] = {
    # rap / hip-hop
    "hip hop": "rap/hip-hop", "rap": "rap/hip-hop", "trap": "rap/hip-hop",
    "drill": "rap/hip-hop", "boom bap": "rap/hip-hop",
    "alternative hip hop": "rap/hip-hop", "underground hip hop": "rap/hip-hop",
    "experimental hip hop": "rap/hip-hop", "conscious hip hop": "rap/hip-hop",
    "west coast hip hop": "rap/hip-hop", "east coast hip hop": "rap/hip-hop",
    "industrial hip hop": "rap/hip-hop", "cloud rap": "rap/hip-hop",
    "jazz rap": "rap/hip-hop", "glitch hop": "rap/hip-hop",
    "g funk": "rap/hip-hop", "horrorcore": "rap/hip-hop",
    # r&b / soul
    "soul": "r&b/soul", "neo soul": "r&b/soul", "rnb": "r&b/soul",
    "r&b": "r&b/soul", "r n b": "r&b/soul", "alternative rnb": "r&b/soul",
    "alternative r&b": "r&b/soul", "motown": "r&b/soul", "gospel": "r&b/soul",
    "doo wop": "r&b/soul",
    # indie (checked before rock so 'indie rock' goes here)
    "indie": "indie", "indie rock": "indie", "indie pop": "indie",
    "indie folk": "indie", "lo fi": "indie", "lofi": "indie",
    "bedroom pop": "indie", "dream pop": "indie", "shoegaze": "indie",
    "k indie": "indie", "k rock": "indie",
    # rock
    "rock": "rock", "alt rock": "rock", "alternative": "rock",
    "alternative rock": "rock", "classic rock": "rock", "hard rock": "rock",
    "punk": "rock", "punk rock": "rock", "post punk": "rock",
    "post rock": "rock", "psych rock": "rock", "psychedelic rock": "rock",
    "garage rock": "rock", "rock n roll": "rock", "rockabilly": "rock",
    "grunge": "rock", "emo": "rock", "math rock": "rock",
    "korean rock": "rock", "j rock": "rock", "british": "rock",
    "british invasion": "rock", "experimental": "rock",
    # metal (more specific than rock)
    "metal": "metal", "heavy metal": "metal", "death metal": "metal",
    "black metal": "metal", "doom metal": "metal", "thrash metal": "metal",
    "metalcore": "metal", "nu metal": "metal",
    "industrial rock": "metal", "industrial metal": "metal",
    "grindcore": "metal", "hardcore punk": "metal",
    # 'industrial' alone is weak (could be NIN-style metal, electronic-industrial,
    # or industrial hip-hop — see _WEAK_TAGS). Specific subgenres handle the rest.
    # pop
    "pop": "pop", "synthpop": "pop", "synth pop": "pop", "electropop": "pop",
    "dance pop": "pop", "power pop": "pop", "k pop": "pop", "j pop": "pop",
    "j idol": "pop", "cantopop": "pop", "mandopop": "pop",
    "pop film": "pop", "pop rock": "pop",
    # electronic
    "electronic": "electronic", "edm": "electronic", "house": "electronic",
    "deep house": "electronic", "tech house": "electronic",
    "progressive house": "electronic", "chicago house": "electronic",
    "french house": "electronic", "techno": "electronic",
    "minimal techno": "electronic", "detroit techno": "electronic",
    "acid techno": "electronic", "trance": "electronic",
    "ambient": "electronic", "ambient techno": "electronic",
    "drum and bass": "electronic", "drum n bass": "electronic",
    "dnb": "electronic", "dubstep": "electronic", "breakbeat": "electronic",
    "garage": "electronic", "uk garage": "electronic",
    "trip hop": "electronic", "idm": "electronic", "glitch": "electronic",
    "electro": "electronic", "club": "electronic", "dance": "electronic",
    "downtempo": "electronic", "j dance": "electronic",
    "vaporwave": "electronic", "chillwave": "electronic",
    "future bass": "electronic", "hardstyle": "electronic",
    # jazz
    "jazz": "jazz", "smooth jazz": "jazz", "free jazz": "jazz",
    "bebop": "jazz", "swing": "jazz", "big band": "jazz",
    "fusion": "jazz", "jazz fusion": "jazz", "bossa nova": "jazz",
    # classical
    "classical": "classical", "baroque": "classical", "romantic": "classical",
    "opera": "classical", "orchestral": "classical", "piano": "classical",
    "modern classical": "classical", "minimalism": "classical",
    "soundtrack": "classical", "film score": "classical", "score": "classical",
    "show tunes": "classical", "new age": "classical",
    # folk / country
    "folk": "folk/country", "folk rock": "folk/country",
    "country": "folk/country", "americana": "folk/country",
    "bluegrass": "folk/country", "honky tonk": "folk/country",
    "singer songwriter": "folk/country", "songwriter": "folk/country",
    "acoustic": "folk/country", "country rock": "folk/country",
    "outlaw country": "folk/country", "alt country": "folk/country",
    # latin
    "latin": "latin", "latino": "latin", "reggaeton": "latin",
    "salsa": "latin", "samba": "latin", "bachata": "latin",
    "mariachi": "latin", "tango": "latin", "forro": "latin",
    "mpb": "latin", "pagode": "latin", "sertanejo": "latin",
    "spanish": "latin", "brazil": "latin",
    # world
    "world music": "world", "world": "world", "afrobeat": "world",
    "afrobeats": "world", "reggae": "world", "ska": "world", "dub": "world",
    "dancehall": "world", "anime": "world", "indian": "world",
    "iranian": "world", "turkish": "world", "german": "world",
    "french": "world", "swedish": "world", "malay": "world",
    "highlife": "world", "soukous": "world",
    # funk / disco
    "funk": "funk/disco", "disco": "funk/disco", "p funk": "funk/disco",
    "groove": "funk/disco", "nu disco": "funk/disco", "boogie": "funk/disco",
    "funk rock": "funk/disco",
    # blues
    "blues": "rock", "blues rock": "rock", "delta blues": "rock",
}


def bucket_for_genres(genres: list[str], default: str = "other") -> str:
    """Walk an artist's merged genres in priority order; first non-weak
    match wins. If only weak tags appear, fall back to the first weak match.

    This handles cases like JPEGMAFIA ['industrial', 'hip hop', ...]:
    'industrial' is weak so we keep scanning, hit 'hip hop' (specific),
    return rap/hip-hop. For Aphex Twin ['ambient', 'electronic', ...]
    'ambient' is specific so it wins immediately.
    """
    weak_fallback: str | None = None
    for g in genres:
        b = _GENRE_TO_BUCKET.get(g)
        if b is None:
            continue
        if g in _WEAK_TAGS:
            if weak_fallback is None:
                weak_fallback = b
            continue
        return b
    return weak_fallback if weak_fallback is not None else default


def color_for_bucket(bucket: str) -> str:
    return GENRE_BUCKETS.get(bucket, GENRE_BUCKETS["other"])


def all_buckets() -> list[str]:
    """Stable bucket order for legends."""
    return list(GENRE_BUCKETS.keys())


def bucket_for_artist(genres: list[str]) -> tuple[str, str]:
    """Convenience: returns (bucket, color)."""
    b = bucket_for_genres(genres)
    return b, color_for_bucket(b)
