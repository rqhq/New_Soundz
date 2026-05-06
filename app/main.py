"""Streamlit entrypoint.

Run from repo root with:

    uv run streamlit run app/main.py --server.port=5000

The Spotify dashboard's redirect URI must match the URL Streamlit serves on
exactly. We use port 5000 to reuse the existing http://127.0.0.1:5000/callback
registration. Streamlit serves any path under its root, so /callback works.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TIME_RANGE_LABELS = {
    "short_term": "Last 4 weeks",
    "medium_term": "Last 6 months",
    "long_term": "All time",
}

SPOTIFY_GREEN = "#1DB954"
DARK_BG = "rgba(0,0,0,0)"

_PLOTLY_DARK = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=DARK_BG,
    font=dict(color="#FFFFFF", family="sans-serif"),
)


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

    if st.session_state.get("authenticated") or auth.validate_token(
        auth.cache_handler.get_cached_token()
    ):
        st.session_state.authenticated = True
        return spotipy.Spotify(auth_manager=auth)

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

    auth_url = auth.get_authorize_url()
    st.title("Spotify Recommender")
    st.write("Personal artist recommendations from your Spotify listening history.")
    st.markdown(f"[**Connect Spotify →**]({auth_url})")
    st.caption(
        "This app is in Spotify dev mode (max 25 allowlisted users). "
        "If you can't log in, you're not on the allowlist yet."
    )
    return None


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_top_artists(_sp: spotipy.Spotify, time_range: str, limit: int = 50) -> list[dict]:
    return _sp.current_user_top_artists(limit=limit, time_range=time_range)["items"]


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_top_tracks(_sp: spotipy.Spotify, time_range: str, limit: int = 50) -> list[dict]:
    return _sp.current_user_top_tracks(limit=limit, time_range=time_range)["items"]


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_recently_played(_sp: spotipy.Spotify, limit: int = 50) -> list[dict]:
    return _sp.current_user_recently_played(limit=limit)["items"]


N_RECS = 20
TOP_LIMIT = 50
UMAP_BACKGROUND_SAMPLE = 1500


@st.cache_resource(show_spinner="Loading recommender model...")
def _load_recsys():
    from spotify_recs.recommender import load_pipeline
    from spotify_recs.routing import build_norm_to_id

    pipe = load_pipeline()
    norm_to_id = build_norm_to_id()
    lookup = pd.read_parquet(REPO_ROOT / "data/processed/artist_lookup.parquet")
    id_to_canonical = dict(
        zip(lookup["artist_id"].astype(int), lookup["canonical_name"])
    )
    return pipe, norm_to_id, id_to_canonical


@st.cache_resource(show_spinner=False)
def _umap_background_embeddings():
    """Sample popular CF artist embeddings to use as UMAP background. Cached."""
    pipe, _, _ = _load_recsys()
    scorer = pipe.node("scorer").component
    embeddings = scorer.item_embeddings
    item_ids = scorer.items.ids()

    norms = np.linalg.norm(embeddings, axis=1)
    top_idx = np.argsort(norms)[-UMAP_BACKGROUND_SAMPLE:]
    bg_emb = embeddings[top_idx].copy()
    bg_ids = item_ids[top_idx].copy()

    lookup = pd.read_parquet(REPO_ROOT / "data/processed/artist_lookup.parquet")
    name_by_id = dict(zip(lookup["artist_id"].astype(int), lookup["canonical_name"]))
    bg_names = [name_by_id.get(int(i), "?") for i in bg_ids]
    return bg_emb, bg_ids, bg_names


def _umap_project_with_user(seed_ids: list[int], rec_ids: list[int]) -> dict:
    """Single fit_transform over background + user points. Returns coord dict."""
    import umap

    pipe, _, _ = _load_recsys()
    scorer = pipe.node("scorer").component
    embeddings = scorer.item_embeddings
    items_vocab = scorer.items

    bg_emb, bg_ids, bg_names = _umap_background_embeddings()
    n_bg = len(bg_emb)

    seed_rows = items_vocab.numbers(seed_ids) if seed_ids else np.empty(0, dtype=int)
    rec_rows = items_vocab.numbers(rec_ids) if rec_ids else np.empty(0, dtype=int)
    seed_emb = embeddings[seed_rows] if len(seed_rows) else np.empty((0, embeddings.shape[1]))
    rec_emb = embeddings[rec_rows] if len(rec_rows) else np.empty((0, embeddings.shape[1]))

    stacked = np.vstack([bg_emb, seed_emb, rec_emb])
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.15, random_state=42, metric="cosine")
    coords = reducer.fit_transform(stacked)

    bg_coords = coords[:n_bg]
    seed_coords = coords[n_bg : n_bg + len(seed_rows)]
    rec_coords = coords[n_bg + len(seed_rows) :]
    return {
        "bg_x": bg_coords[:, 0], "bg_y": bg_coords[:, 1], "bg_names": bg_names,
        "seed_x": seed_coords[:, 0], "seed_y": seed_coords[:, 1],
        "rec_x": rec_coords[:, 0], "rec_y": rec_coords[:, 1],
    }


@st.cache_data(ttl=3600, show_spinner=False)
def _compute_recs(user_id: str, short_artists: tuple, medium_artists: tuple) -> dict:
    """Run the full route → fold-in → expand pipeline. Cached per user."""
    from spotify_recs.align import normalize_artist
    from spotify_recs.cache import ArtistCache
    from spotify_recs.recommender import recommend_for_history
    from spotify_recs.routing import expand_to_modern, route_artists

    pipe, norm_to_id, _ = _load_recsys()

    short_dicts = [{"name": n} for n in short_artists]
    medium_dicts = [{"name": n} for n in medium_artists]

    with ArtistCache() as cache:
        result = route_artists(short_dicts, medium_dicts, cache=cache, norm_to_id=norm_to_id)

    if len(result.weights) < 5:
        return {"sparse": True, "n_routed": len(result.weights)}

    classic = recommend_for_history(pipe, result.weights, n=N_RECS)
    expanded = recommend_for_history(pipe, result.weights, n=50)
    user_norms = {normalize_artist(n) for n in (short_artists + medium_artists)}
    with ArtistCache() as cache:
        modern = expand_to_modern(
            expanded, cache=cache, norm_to_id=norm_to_id,
            exclude_norms=user_norms, n=N_RECS,
        )

    return {
        "sparse": False,
        "weights": result.weights,
        "classic": classic,
        "modern": modern,
        "n_direct": result.n_direct,
        "n_proxy_only": result.n_proxy_only,
        "n_unmatched": len(result.unmatched),
    }


def _now_playing_strip(sp: spotipy.Spotify) -> None:
    try:
        playing = sp.current_user_playing_track()
    except Exception:
        playing = None

    if not playing or not playing.get("item"):
        st.info("Nothing currently playing.")
        return

    item = playing["item"]
    artists = ", ".join(a["name"] for a in item["artists"])
    progress_ms = playing.get("progress_ms") or 0
    duration_ms = item.get("duration_ms") or 1
    pct = min(progress_ms / duration_ms, 1.0)

    cols = st.columns([1, 6])
    with cols[0]:
        if item["album"].get("images"):
            st.image(item["album"]["images"][-1]["url"], width=80)
    with cols[1]:
        status = "▶ Now playing" if playing.get("is_playing") else "⏸ Paused"
        st.markdown(f"**{status}** — {item['name']} · {artists}")
        st.caption(item["album"]["name"])
        st.progress(pct)


def _hero_header(sp: spotipy.Spotify, me: dict) -> None:
    avatar_url = me["images"][0]["url"] if me.get("images") else None
    followers = me.get("followers", {}).get("total", 0)

    cols = st.columns([1, 6])
    with cols[0]:
        if avatar_url:
            st.markdown(
                f'<img src="{avatar_url}" '
                f'style="width:120px;height:120px;border-radius:50%;object-fit:cover;'
                f'box-shadow:0 4px 16px rgba(0,0,0,0.5);">',
                unsafe_allow_html=True,
            )
    with cols[1]:
        st.markdown(
            f'<div style="font-size:0.85rem;color:#B3B3B3;letter-spacing:0.1em;'
            f'text-transform:uppercase;margin-bottom:0.25rem;">Profile</div>'
            f'<div style="font-size:3rem;font-weight:700;line-height:1.1;">'
            f'{me["display_name"]}</div>'
            f'<div style="color:#B3B3B3;margin-top:0.5rem;">'
            f'{followers:,} followers · @{me["id"]}</div>',
            unsafe_allow_html=True,
        )


def _artist_card_row(artists: list[dict], n: int = 8) -> None:
    """Horizontal row of large artist cards: image + name + top genres."""
    artists = artists[:n]
    cols = st.columns(min(n, len(artists)))
    for col, a in zip(cols, artists):
        img = a["images"][0]["url"] if a.get("images") else ""
        genre = a["genres"][0] if a.get("genres") else ""
        with col:
            if img:
                st.markdown(
                    f'<a href="{a["external_urls"]["spotify"]}" target="_blank" '
                    f'style="text-decoration:none;color:inherit;">'
                    f'<img src="{img}" '
                    f'style="width:100%;aspect-ratio:1/1;object-fit:cover;'
                    f'border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.4);">'
                    f'<div style="margin-top:0.6rem;font-weight:600;font-size:0.95rem;">'
                    f'{a["name"]}</div>'
                    f'<div style="color:#B3B3B3;font-size:0.8rem;margin-top:0.15rem;">'
                    f'{genre}</div>'
                    f'</a>',
                    unsafe_allow_html=True,
                )


def _render_overview(sp: spotipy.Spotify) -> None:
    me = sp.current_user()
    _hero_header(sp, me)

    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:1.4rem;font-weight:700;margin-bottom:0.5rem;">'
        'Now playing</div>',
        unsafe_allow_html=True,
    )
    _now_playing_strip(sp)

    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:1.4rem;font-weight:700;margin-bottom:1rem;">'
        'Your top artists this month</div>',
        unsafe_allow_html=True,
    )
    top = _fetch_top_artists(sp, "short_term", limit=8)
    _artist_card_row(top, n=8)


def _kpi_row(sp: spotipy.Spotify) -> None:
    short_artists = _fetch_top_artists(sp, "short_term", limit=50)
    long_artists = _fetch_top_artists(sp, "long_term", limit=50)
    recent = _fetch_recently_played(sp, limit=50)

    short_names = {a["name"] for a in short_artists}
    long_names = {a["name"] for a in long_artists}
    new_entrants = len(short_names - long_names)

    genre_counter: Counter[str] = Counter()
    for a in short_artists:
        for g in a.get("genres", []):
            genre_counter[g] += 1
    top_genre = genre_counter.most_common(1)[0][0] if genre_counter else "—"

    unique_recent_artists = len({
        item["track"]["artists"][0]["name"] for item in recent if item.get("track")
    })

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top artists tracked (4w)", len(short_artists))
    c2.metric("New entrants vs all-time", new_entrants)
    c3.metric("Dominant genre (4w)", top_genre)
    c4.metric("Unique artists in last 50 plays", unique_recent_artists)


def _top_tracks_tabs(sp: spotipy.Spotify) -> None:
    st.subheader("Top tracks")
    tabs = st.tabs(list(TIME_RANGE_LABELS.values()))
    for tab, (range_key, _label) in zip(tabs, TIME_RANGE_LABELS.items()):
        with tab:
            tracks = _fetch_top_tracks(sp, range_key, limit=20)
            if not tracks:
                st.write("No data for this range.")
                continue
            rows = []
            for i, t in enumerate(tracks, 1):
                images = t["album"].get("images") or []
                rows.append({
                    "#": i,
                    "Cover": images[-1]["url"] if images else None,
                    "Track": t["name"],
                    "Artist": ", ".join(a["name"] for a in t["artists"]),
                    "Album": t["album"]["name"],
                    "Popularity": t.get("popularity", 0),
                })
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Cover": st.column_config.ImageColumn("", width="small"),
                    "Popularity": st.column_config.ProgressColumn(
                        "Popularity", min_value=0, max_value=100, format="%d"
                    ),
                },
            )


def _top_genres_bar(sp: spotipy.Spotify) -> None:
    st.subheader("Top genres (from your top 50 artists, last 6 months)")
    artists = _fetch_top_artists(sp, "medium_term", limit=50)
    counter: Counter[str] = Counter()
    for a in artists:
        for g in a.get("genres", []):
            counter[g] += 1
    if not counter:
        st.caption("Spotify returned no genre tags for these artists.")
        return
    top = counter.most_common(15)
    df = pd.DataFrame(top, columns=["genre", "count"]).iloc[::-1]
    fig = px.bar(
        df, x="count", y="genre", orientation="h", height=420,
        color_discrete_sequence=[SPOTIFY_GREEN],
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title=None, xaxis_title="Artists",
        **_PLOTLY_DARK,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    st.plotly_chart(fig, use_container_width=True)


def _artist_movement_table(sp: spotipy.Spotify) -> None:
    st.subheader("Artist movement: short-term vs medium-term")
    short = _fetch_top_artists(sp, "short_term", limit=50)
    medium = _fetch_top_artists(sp, "medium_term", limit=50)
    short_rank = {a["name"]: i + 1 for i, a in enumerate(short)}
    medium_rank = {a["name"]: i + 1 for i, a in enumerate(medium)}

    rows = []
    for name, s_rank in short_rank.items():
        m_rank = medium_rank.get(name)
        if m_rank is None:
            delta_label = "🆕 new"
        else:
            delta = m_rank - s_rank
            if delta == 0:
                delta_label = "—"
            elif delta > 0:
                delta_label = f"▲ +{delta}"
            else:
                delta_label = f"▼ {delta}"
        rows.append({
            "Artist": name,
            "4w rank": s_rank,
            "6mo rank": m_rank if m_rank else "—",
            "Movement": delta_label,
        })
    st.dataframe(pd.DataFrame(rows[:20]), hide_index=True, use_container_width=True)


def _interleave(classic: list[dict], modern: list[dict], n: int) -> list[dict]:
    """Round-robin merge two ranked lists. Stops at n total."""
    out = []
    i = 0
    while len(out) < n and (i < len(classic) or i < len(modern)):
        if i < len(classic):
            out.append(classic[i])
            if len(out) >= n:
                break
        if i < len(modern):
            out.append(modern[i])
        i += 1
    return out


def _rec_list(items: list[dict]) -> None:
    if not items:
        st.info("No recommendations to show in this list.")
        return
    rows = [
        {
            "#": i,
            "Artist": it["name"],
            "Source": it["source"],
            "Score": round(it["score"], 3),
        }
        for i, it in enumerate(items, 1)
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_umap(user_id: str, seed_ids: tuple[int, ...], rec_ids: tuple[int, ...]) -> dict:
    return _umap_project_with_user(list(seed_ids), list(rec_ids))


def _umap_chart(proj: dict, seed_names: list[str], rec_names: list[str]) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=proj["bg_x"], y=proj["bg_y"],
        mode="markers",
        marker=dict(size=4, color="rgba(150,150,150,0.25)"),
        hovertext=proj["bg_names"],
        hoverinfo="text",
        name="CF vocab",
    ))
    if len(proj["seed_x"]):
        fig.add_trace(go.Scatter(
            x=proj["seed_x"], y=proj["seed_y"],
            mode="markers",
            marker=dict(size=12, color="#1DB954", line=dict(width=1, color="white")),
            hovertext=seed_names,
            hoverinfo="text",
            name="Your seeds",
        ))
    if len(proj["rec_x"]):
        fig.add_trace(go.Scatter(
            x=proj["rec_x"], y=proj["rec_y"],
            mode="markers",
            marker=dict(size=14, color="#F59E0B", symbol="star", line=dict(width=1, color="white")),
            hovertext=rec_names,
            hoverinfo="text",
            name="Classic recs",
        ))
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        legend=dict(orientation="h", y=-0.05),
        **_PLOTLY_DARK,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"UMAP projection of the top {UMAP_BACKGROUND_SAMPLE} CF artists plus your seeds and recs, "
        "fit jointly in cosine space. Modern recs aren't shown — they're outside the CF vocab and "
        "have no embedding."
    )


NODE_STYLE = {
    "seed": {
        "color": "#1DB954", 
        "halo": "rgba(29, 185, 84, 0.27)", 
        "label": "Your seeds"
    },
    "classic": {
        "color": "#F59E0B", 
        "halo": "rgba(245, 158, 11, 0.27)", 
        "label": "Classic recs"
    },
    "modern": {
        "color": "#22D3EE", 
        "halo": "rgba(34, 211, 238, 0.27)", 
        "label": "Modern recs"
    },
}


def _build_similarity_network(
    seed_ids: list[int],
    classic_df: pd.DataFrame,
    modern_df: pd.DataFrame,
    weights: dict[int, float],
    top_k_edges: int = 4,
    max_seeds: int = 18,
) -> dict:
    """Assemble nodes + edges for the similarity network.

    CF-CF edges come from ALS cosine similarity (top-k per node).
    Modern→CF edges come from `support_seeds` (Last.fm similarity scores).
    """
    pipe, _, id_to_canonical = _load_recsys()
    scorer = pipe.node("scorer").component
    embeddings = scorer.item_embeddings
    items_vocab = scorer.items

    nodes: list[dict] = []

    seed_pairs = sorted(
        ((aid, weights.get(aid, 0.0)) for aid in seed_ids),
        key=lambda x: -x[1],
    )[:max_seeds]
    for aid, w in seed_pairs:
        nodes.append({
            "key": f"s:{aid}",
            "name": id_to_canonical.get(aid, "?"),
            "type": "seed",
            "value": float(w),
            "item_id": int(aid),
        })

    for r in classic_df.itertuples(index=False):
        nodes.append({
            "key": f"c:{int(r.item_id)}",
            "name": r.canonical_name,
            "type": "classic",
            "value": float(r.score),
            "item_id": int(r.item_id),
        })

    for r in modern_df.itertuples(index=False):
        nodes.append({
            "key": f"m:{r.name}",
            "name": r.name,
            "type": "modern",
            "value": float(r.score),
            "item_id": None,
            "support_seeds": list(r.support_seeds) if hasattr(r, "support_seeds") else [],
        })

    cf_nodes = [n for n in nodes if n["item_id"] is not None]
    edges: list[dict] = []
    seen_pairs: set = set()

    if len(cf_nodes) >= 2:
        cf_keys = [n["key"] for n in cf_nodes]
        cf_ids = [n["item_id"] for n in cf_nodes]
        cf_emb = embeddings[items_vocab.numbers(cf_ids)]
        norms = np.linalg.norm(cf_emb, axis=1, keepdims=True)
        cf_unit = cf_emb / np.clip(norms, 1e-8, None)
        sim = cf_unit @ cf_unit.T
        np.fill_diagonal(sim, -np.inf)

        n = len(cf_keys)
        k = min(top_k_edges, n - 1)
        top_idx = np.argpartition(-sim, k, axis=1)[:, :k]
        for i in range(n):
            for j in top_idx[i]:
                if i == j:
                    continue
                a, b = sorted([cf_keys[i], cf_keys[j]])
                if (a, b) in seen_pairs:
                    continue
                seen_pairs.add((a, b))
                edges.append({"src": a, "dst": b, "weight": float(sim[i, j]), "kind": "als"})

    name_to_key = {n["name"]: n["key"] for n in nodes if n["type"] in ("seed", "classic")}
    for n in nodes:
        if n["type"] != "modern":
            continue
        for seed_name, sim_score in n["support_seeds"]:
            if seed_name in name_to_key:
                edges.append({
                    "src": n["key"], "dst": name_to_key[seed_name],
                    "weight": float(sim_score), "kind": "lastfm",
                })

    return {"nodes": nodes, "edges": edges}


@st.cache_data(ttl=3600, show_spinner=False)
def _network_layout(net_signature: tuple) -> dict:
    """Force-directed layout. Cached because we re-render the chart on filter changes.

    `net_signature` is a stable tuple representation of the network (keys + edges).
    """
    import networkx as nx

    keys, edges = net_signature
    G = nx.Graph()
    G.add_nodes_from(keys)
    for src, dst, w in edges:
        # Convert similarity (high = close) to spring force; weight pulls nodes together.
        G.add_edge(src, dst, weight=max(0.05, w))

    pos = nx.spring_layout(G, k=0.9, iterations=200, seed=42, weight="weight")
    return {k: (float(p[0]), float(p[1])) for k, p in pos.items()}


def _render_similarity_network(net: dict, n_labels: int = 18) -> None:
    """Render the network with Plotly: edges as lines, nodes as glowing markers."""
    nodes = net["nodes"]
    edges = net["edges"]
    if not nodes:
        st.caption("No nodes to display.")
        return

    sig_edges = tuple(sorted((e["src"], e["dst"], round(e["weight"], 4)) for e in edges))
    sig_keys = tuple(sorted(n["key"] for n in nodes))
    pos = _network_layout((sig_keys, sig_edges))

    fig = go.Figure()

    if edges:
        max_w = max(e["weight"] for e in edges) or 1.0
        for kind, color in (("als", "rgba(255,255,255,0.18)"), ("lastfm", "rgba(34,211,238,0.35)")):
            xs, ys, widths = [], [], []
            for e in edges:
                if e["kind"] != kind:
                    continue
                if e["src"] not in pos or e["dst"] not in pos:
                    continue
                x0, y0 = pos[e["src"]]
                x1, y1 = pos[e["dst"]]
                xs.extend([x0, x1, None])
                ys.extend([y0, y1, None])
                widths.append(0.6 + 2.2 * (e["weight"] / max_w))
            if not xs:
                continue
            avg_w = sum(widths) / len(widths) if widths else 1.0
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=color, width=avg_w),
                hoverinfo="skip", showlegend=False,
            ))

    by_type: dict[str, list[dict]] = {"seed": [], "classic": [], "modern": []}
    for n in nodes:
        by_type.setdefault(n["type"], []).append(n)

    for ntype, items in by_type.items():
        if not items:
            continue
        style = NODE_STYLE[ntype]
        vals = [n["value"] for n in items]
        vmin, vmax = min(vals), max(vals)
        vrange = (vmax - vmin) or 1.0

        sizes = [22 + 30 * ((v - vmin) / vrange) for v in vals]
        xs = [pos[n["key"]][0] for n in items if n["key"] in pos]
        ys = [pos[n["key"]][1] for n in items if n["key"] in pos]
        names = [n["name"] for n in items if n["key"] in pos]

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(
                size=[s + 14 for s in sizes],
                color=style["halo"],
                line=dict(width=0),
            ),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(
                size=sizes,
                color=style["color"],
                line=dict(width=2, color="rgba(255,255,255,0.85)"),
            ),
            hovertext=names,
            hoverinfo="text",
            name=style["label"],
        ))

    top_for_labels = sorted(nodes, key=lambda n: -n["value"])[:n_labels]
    label_xs, label_ys, label_text = [], [], []
    for n in top_for_labels:
        if n["key"] not in pos:
            continue
        x, y = pos[n["key"]]
        label_xs.append(x)
        label_ys.append(y)
        label_text.append(n["name"])
    if label_xs:
        fig.add_trace(go.Scatter(
            x=label_xs, y=label_ys, mode="text",
            text=label_text,
            textposition="top center",
            textfont=dict(color="rgba(255,255,255,0.92)", size=12, family="sans-serif"),
            hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(
        height=720,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        legend=dict(
            orientation="h", y=-0.02,
            font=dict(color="#FFFFFF"),
            bgcolor="rgba(0,0,0,0)",
        ),
        **_PLOTLY_DARK,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Edges between CF artists (white) reflect cosine similarity in the ALS embedding space. "
        "Edges from modern recs (cyan) reflect Last.fm similarity to your classic recs — these "
        "are exactly the support links the reverse-proxy uses to surface post-2009 artists."
    )


def _render_recommendations(sp: spotipy.Spotify) -> None:
    st.title("Recommendations")

    me = sp.current_user()
    short = _fetch_top_artists(sp, "short_term", limit=TOP_LIMIT)
    medium = _fetch_top_artists(sp, "medium_term", limit=TOP_LIMIT)
    short_names = tuple(a["name"] for a in short)
    medium_names = tuple(a["name"] for a in medium)

    with st.spinner("Routing your top artists and running fold-in inference..."):
        recs = _compute_recs(me["id"], short_names, medium_names)

    if recs.get("sparse"):
        st.warning(
            f"Only {recs['n_routed']} of your top artists could be matched into the CF vocab. "
            "The recommender needs at least 5 to produce trustworthy results. "
            "Content-based fallback isn't built yet (Day 4 step 4)."
        )
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Direct CF matches", recs["n_direct"])
    c2.metric("Proxy-only artists", recs["n_proxy_only"])
    c3.metric("Unmatched", recs["n_unmatched"])

    st.divider()

    classic = recs["classic"]
    modern = recs["modern"]

    classic_items = [
        {"name": r.canonical_name, "score": float(r.score), "source": "classic"}
        for r in classic.itertuples(index=False)
    ]
    modern_items = [
        {"name": r.name, "score": float(r.score), "source": "modern"}
        for r in modern.itertuples(index=False)
    ] if not modern.empty else []

    mode = st.radio(
        "Rec mix",
        ["Mixed", "Classic", "Modern"],
        horizontal=True,
        help=(
            "Classic = direct ALS recs (bounded by Last.fm 360K's 2008 vocab). "
            "Modern = post-2009 candidates surfaced via Last.fm similars from the classic recs. "
            "Mixed interleaves them."
        ),
    )
    if mode == "Mixed":
        items = _interleave(classic_items, modern_items, n=N_RECS)
    elif mode == "Classic":
        items = classic_items[:N_RECS]
    else:
        items = modern_items[:N_RECS]

    _rec_list(items)

    st.divider()
    st.subheader("Embedding-space map")

    seed_ids = tuple(int(aid) for aid in recs["weights"].keys())
    rec_ids = tuple(int(r.item_id) for r in classic.itertuples(index=False))
    _, _, id_to_canonical = _load_recsys()
    seed_names = [id_to_canonical.get(i, "?") for i in seed_ids]
    rec_names = [r.canonical_name for r in classic.itertuples(index=False)]

    with st.spinner("Fitting UMAP projection (~10s, cached after first run)..."):
        proj = _cached_umap(me["id"], seed_ids, rec_ids)
    _umap_chart(proj, seed_names, rec_names)

    st.divider()
    st.subheader("Artist similarity network")
    with st.spinner("Building similarity network..."):
        net = _build_similarity_network(
            seed_ids=list(recs["weights"].keys()),
            classic_df=classic,
            modern_df=modern,
            weights=recs["weights"],
        )
    _render_similarity_network(net)


def _render_analytics(sp: spotipy.Spotify) -> None:
    st.title("Listening analytics")
    _kpi_row(sp)
    st.divider()
    left, right = st.columns([1, 1])
    with left:
        _top_genres_bar(sp)
    with right:
        _artist_movement_table(sp)
    st.divider()
    _top_tracks_tabs(sp)


def main() -> None:
    sp = _get_authenticated_client()
    if sp is None:
        return

    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio(
            "Page",
            ["Overview", "Analytics", "Recommendations"],
            label_visibility="collapsed",
        )
        st.divider()
        if st.button("Log out"):
            for key in ("auth_manager", "authenticated"):
                st.session_state.pop(key, None)
            TOKEN_CACHE_PATH.unlink(missing_ok=True)
            st.cache_data.clear()
            st.rerun()

    if page == "Overview":
        _render_overview(sp)
    elif page == "Analytics":
        _render_analytics(sp)
    else:
        _render_recommendations(sp)


if __name__ == "__main__":
    main()
