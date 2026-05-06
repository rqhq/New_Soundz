"""Train ALS on Last.fm 360K, persist the pipeline, serve recs for new users.

Day 3 of the project. Two paths through this module:

  1. **Training (offline, once)** — `train_and_save()` reads
     `data/processed/interactions.parquet`, builds a LensKit `Dataset`, fits
     `ImplicitMFScorer`, and pickles the trained pipeline to `models/als.pkl`.

  2. **Inference (online, per request)** — `recommend_for_history(pipeline,
     artist_weights, n=20)` takes a dict of `{artist_id: weight}` and returns
     a top-N artist DataFrame. This is the fold-in path used by the eventual
     webapp; it does *not* require the user to be in the training data.

LensKit 2025.1+ pipeline API. The 0.x scikit-style API is gone — anything you
remember from before that is wrong.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lenskit import Pipeline, recommend, topn_pipeline
from lenskit.als import ImplicitMFScorer
from lenskit.data import ItemList, RecQuery, from_interactions_df

INTERACTIONS_PATH = Path("data/processed/interactions.parquet")
ARTIST_LOOKUP_PATH = Path("data/processed/artist_lookup.parquet")
MODEL_PATH = Path("models/als.pkl")

# Last.fm uses these as catch-all bins for unidentified or compilation tracks.
# They have huge play counts and dominate ALS recommendations otherwise.
JUNK_NORM_NAMES = frozenset({
    "[unknown]", "unknown", "unknown artist",
    "various artists", "various", "va",
    "soundtrack", "ost", "original soundtrack",
    "anonymous", "traditional",
})

# ALS hyperparameters — defaults from LensKit, untuned. Day 6 candidate for tuning.
EMBEDDING_SIZE = 64
EPOCHS = 10
REGULARIZATION = 0.1
CONFIDENCE_WEIGHT = 40  # alpha in c_ui = 1 + alpha * count


def _load_dataset() -> tuple[pd.DataFrame, "Dataset"]:
    """Load interactions and wrap them as a LensKit Dataset.

    Renames artist_id → item_id and count → rating because that's what
    `from_interactions_df` looks for. ImplicitMFScorer reads `rating` as the
    confidence-weighting input (despite the name) when `use_ratings=False` —
    the count column ends up as `c_ui = 1 + alpha * count` internally.
    """
    df = pd.read_parquet(INTERACTIONS_PATH)
    df = df.rename(columns={"artist_id": "item_id", "count": "rating"})
    print(f"  loaded {len(df):,} interactions, "
          f"{df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items")
    ds = from_interactions_df(df)
    return df, ds


def train_and_save(
    embedding_size: int = EMBEDDING_SIZE,
    epochs: int = EPOCHS,
    regularization: float = REGULARIZATION,
    confidence_weight: float = CONFIDENCE_WEIGHT,
    out_path: Path = MODEL_PATH,
) -> Pipeline:
    """Train ALS on the full Last.fm + me interaction matrix, save to disk."""
    print("Loading dataset ...")
    _, ds = _load_dataset()

    print(f"\nTraining ImplicitMFScorer "
          f"(embedding_size={embedding_size}, epochs={epochs}, "
          f"reg={regularization}, weight={confidence_weight}) ...")
    scorer = ImplicitMFScorer(
        embedding_size=embedding_size,
        epochs=epochs,
        regularization=regularization,
        weight=confidence_weight,
    )
    pipe = topn_pipeline(scorer, n=200)
    pipe.train(ds)
    print("  trained")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(pipe, f)
    print(f"  saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    return pipe


def load_pipeline(path: Path = MODEL_PATH) -> Pipeline:
    """Load the trained pipeline from disk."""
    with path.open("rb") as f:
        return pickle.load(f)


def recommend_for_user(
    pipeline: Pipeline,
    user_id: str,
    n: int = 20,
) -> pd.DataFrame:
    """Top-N recs for a user already in the training data."""
    items: ItemList = recommend(pipeline, user_id, n=n)
    return _itemlist_to_df(items)


def recommend_for_history(
    pipeline: Pipeline,
    artist_weights: dict[int, float],
    n: int = 20,
    exclude_seen: bool = True,
    exclude_extra: set[int] | None = None,
    popularity_alpha: float = 0.0,
    mmr_lambda: float = 1.0,
) -> pd.DataFrame:
    """Top-N recs for a new (cold-start) user via fold-in inference.

    `artist_weights`: dict of {artist_id: weight}. Weights are passed via the
    `rating` field on the ItemList — LensKit's fold-in only reads them when
    `use_ratings=True` on the scorer config (we flip it here for inference).

    `exclude_seen`: drop the user's own input artists from the result.
    `exclude_extra`: additional artist_ids to drop (e.g., long_term top artists
        the user listens to but that don't contribute to the user vector).

    `popularity_alpha`: dampens ALS popularity bias by dividing each candidate's
        score by `||item_emb||^alpha`. 0 = no dampening (raw ALS). 1 = pure
        cosine (popularity-blind). 0.5 is a reasonable middle. Pulls a wider
        candidate pool (200) before re-ranking so dampening can surface items
        that were buried in the raw ranking.

    `mmr_lambda`: Maximal Marginal Relevance trade-off. 1.0 = pure relevance
        (no diversity step). <1.0 enables greedy MMR re-ranking: each pick
        maximizes `λ·relevance − (1−λ)·max_cosine_to_already_picked` in ALS
        embedding space. 0.7 is a typical "diverse but still on-target" pick;
        0.5 is aggressively diverse.
    """
    item_ids = list(artist_weights.keys())
    weights = [float(artist_weights[i]) for i in item_ids]
    history = ItemList(item_ids=item_ids, rating=weights, ordered=False)
    query = RecQuery(history_items=history)

    scorer = pipeline.node("scorer").component
    scorer.config.use_ratings = True

    # When dampening or MMR-reranking, pull a wide pool so re-ranking has
    # room to lift previously-buried candidates above the popular ones and
    # MMR has enough material to diversify across.
    needs_wide_pool = popularity_alpha > 0 or mmr_lambda < 1.0
    pool_size = 200 if needs_wide_pool else (n + (len(item_ids) if exclude_seen else 0) + len(exclude_extra or set()))
    items: ItemList = recommend(pipeline, query, n=pool_size)
    df = _itemlist_to_df(items)

    if popularity_alpha > 0 and len(df):
        embeddings = scorer.item_embeddings
        items_vocab = scorer.items
        rows = items_vocab.numbers(df["item_id"].tolist())
        norms = np.linalg.norm(embeddings[rows], axis=1)
        df = df.assign(score=df["score"] / np.maximum(norms ** popularity_alpha, 1e-8))
        df = df.sort_values("score", ascending=False).reset_index(drop=True)

    if mmr_lambda < 1.0 and len(df) > 1:
        df = _mmr_rerank(df, scorer, mmr_lambda, n_select=min(n * 3, len(df)))

    extra = exclude_extra or set()
    drop_ids: set[int] = set(extra)
    if exclude_seen:
        drop_ids.update(item_ids)
    if drop_ids:
        df = df[~df["item_id"].isin(drop_ids)]
    return df.head(n).reset_index(drop=True)


def _mmr_rerank(
    df: pd.DataFrame,
    scorer,
    mmr_lambda: float,
    n_select: int,
) -> pd.DataFrame:
    """Greedy MMR over `df["score"]` using ALS-cosine for similarity.

    Returns a re-ordered DataFrame of length `min(n_select, len(df))`. The
    ordering is the MMR pick order, so taking `.head(n)` after exclusions
    yields the diverse-relevant top-N.
    """
    items_vocab = scorer.items
    pool_ids = df["item_id"].tolist()
    rows = items_vocab.numbers(pool_ids)
    pool_emb = scorer.item_embeddings[rows]
    norms = np.linalg.norm(pool_emb, axis=1, keepdims=True)
    pool_unit = pool_emb / np.clip(norms, 1e-8, None)

    relevance = df["score"].to_numpy()
    selected: list[int] = []
    remaining = list(range(len(df)))

    # First pick is pure-relevance argmax.
    first = int(np.argmax(relevance))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < n_select and remaining:
        rem_arr = np.array(remaining)
        sim_to_picked = pool_unit[rem_arr] @ pool_unit[selected].T
        max_sim = sim_to_picked.max(axis=1)
        mmr_scores = mmr_lambda * relevance[rem_arr] - (1.0 - mmr_lambda) * max_sim
        best_local = int(np.argmax(mmr_scores))
        selected.append(int(rem_arr[best_local]))
        remaining.pop(best_local)

    return df.iloc[selected].reset_index(drop=True)


def _itemlist_to_df(items: ItemList, drop_junk: bool = True) -> pd.DataFrame:
    """Materialize an ItemList into a (item_id, score) DataFrame, joined with names.

    If drop_junk, removes Last.fm's catch-all bin artists ([unknown], various
    artists, soundtrack, ...) which otherwise pollute the top of every result.
    """
    df = items.to_df()
    if "item_id" not in df.columns and "item" in df.columns:
        df = df.rename(columns={"item": "item_id"})

    lookup = pd.read_parquet(ARTIST_LOOKUP_PATH)
    df = df.merge(lookup, left_on="item_id", right_on="artist_id", how="left")
    if drop_junk:
        df = df[~df["canonical_name"].str.lower().isin(JUNK_NORM_NAMES)]
    return df[["item_id", "canonical_name", "score"]].reset_index(drop=True)


if __name__ == "__main__":
    import sys

    if "--load" in sys.argv:
        print(f"Loading existing model from {MODEL_PATH} ...")
        pipe = load_pipeline()
    else:
        pipe = train_and_save()

    print("\n=== Smoke test 1: recommend for 'me' (known user, junk filtered) ===")
    recs = recommend_for_user(pipe, "me", n=20)
    print(recs.head(20).to_string(index=False))

    print("\n=== Smoke test 2: cold-start fold-in for a synthetic user ===")
    # Build a fake history from a few of my matched artists. Tests that fold-in
    # works without the user being in the training data.
    lookup = pd.read_parquet(ARTIST_LOOKUP_PATH)
    name_to_id = dict(zip(lookup["canonical_name"], lookup["artist_id"]))
    history_names = ["kanye west", "daft punk", "underworld", "d'angelo", "freddie gibbs"]
    history = {name_to_id[n]: 100.0 for n in history_names if n in name_to_id}
    print(f"  fake-user history: {dict((lookup.loc[lookup.artist_id==i,'canonical_name'].iat[0], w) for i,w in history.items())}")
    recs = recommend_for_history(pipe, history, n=20)
    print(recs.head(20).to_string(index=False))
