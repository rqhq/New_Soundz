# New Soundz: A Music Recommender Built Around Its Constraints

## TL;DR

A web app that takes your Spotify listening, runs it through a collaborative-filtering model, and recommends artists you've may have never heard. The headline feature: two interactive sliders that let listeners trade off relevance against *diversity* and *popularity*, making the recommender's bias toward under-explored music an explicit, user-controllable choice rather than a hidden default.

## The Problem

Most music recommenders push you toward what's popular and what's adjacent. Spotify's own playlist surfaces are heavily weighted toward the last decade and toward stuff that already has momentum, which is fine for casual listening but leaves a lot of music buried. The goal of this project was to build something with the opposite gravity: a recommender that, by design, surfaces music you'd otherwise miss.

## The Constraint That Shaped Everything

To do collaborative filtering at any reasonable quality, you need a large dataset of who-listens-to-whom. The only public dataset of that scale is **Last.fm 360K**: a frozen snapshot from 2008–2009 capturing 358,872 users and 267,739 artists. I could not find an accessible modern public equivalent. Spotify, Apple Music, and YouTube do not release this kind of data.

That dataset's age is the project's central design problem. Half the artists a modern listener may care about (i.e. Kendrick Lamar, Frank Ocean, Tyler the Creator, anyone post-2009, etc.),simply don't exist in its vocabulary. Pretending the data was current would have produced a broken recommender. Acknowledging the constraint and designing around it produced a coherent architecture.

## The Architecture: Two Coordinate Systems

The system makes recommendations in two passes, each playing a different role:

**Classic recs** are produced by the CF backbone which is an ALS model trained on Last.fm 360K using fold-in inference. This means the user's Spotify top artists become the seed history and the model returns artists from its (older) vocabulary that listeners with similar taste also enjoyed. These are deliberately *not* current, instead they're catalog surfacing. Think of them as the system's answer to *"what older artists do people who listen to what you listen to also love?"*

**Modern recs** are produced by walking outward from the classic recs through Last.fm's similarity graph. For each older artist the CF model surfaces, the system fetches its similar-artist list, filters out anything already in the CF vocabulary, and ranks the remainder by how strongly multiple classic recs collectively endorse it. The older artists serve as coordinate markers rather than destinations which are proxies that locate where in the musical landscape the listener is. We can identify current artists in the same neighborhood.

Together, the two passes give listeners both kinds of "new": new-to-them deep catalog, and new-period contemporary artists.

## Making the Thesis Tangible

The architecture says "expose listeners to under-listened music," but the recommender's actual scoring still has to enforce that. Two interactive controls make the policy explicit:

**Popularity dampening (α slider).** ALS embeddings inherit a popularity bias — popular artists co-occur with everyone in the training data, so they win raw scoring battles regardless of fit. The slider divides each candidate's score by `‖embedding‖^α`, where α=0 is raw ALS (popularity-favored) and α=1 is pure cosine similarity (popularity-blind). At α=1, the recommender produces noticeably more underground picks; mainstream juggernauts that were dominating the list at α=0 get demoted in favor of artists with smaller, more focused listener bases.

**Diversity (MMR λ slider).** Even with popularity dampened, recommendation lists collapse into clusters: five flavors of the same one artist. Maximal Marginal Relevance re-ranking penalizes each candidate for being too similar (in the same ALS embedding space) to artists already picked. λ=1 is pure relevance; λ=0.7 is "diverse but on-target"; λ=0.5 deliberately spreads picks across clusters. At λ=0.5 with α=1, the list pulls in genuine cross-genre surprises — a Björk or a Justice next to the hip-hop core for a hip-hop-heavy listener.

These aren't internal hyperparameters tucked away in a config file — they're surfaced as live controls. Anyone using the app can dial in their own balance between "sounds-like-what-I-know" and "expose me to something I've never heard."

## Honest Limits

- **The model doesn't learn from feedback.** It's cold-start fold-in over a frozen embedding space, not an online-learning system. The same seed input always produces the same output (modulo the diversity slider).
- **The CF vocabulary is permanently capped at 2008–09.** Any artist who emerged after that is unreachable from the classic-recs path; the modern-recs walk is the only bridge.
- **Modern recs are only as good as Last.fm's similarity graph,** which is crowdsourced and uneven — strong on widely-tagged artists, sparse on niche ones.
- **Per-user streaming-history analytics aren't feasible at multi-tenant scale.** Spotify's API exposes top-artists rankings but not per-track play counts or listening-time totals; getting those requires a 30-day data-export request from each user, which kills the use case.

## What I'd Do With More Time

The most interesting next step is replacing or augmenting the frozen Last.fm 360K backbone with newer co-listening data. Even a smaller yet newer dataset would close the post-2009 gap and let the modern-recs path be a refinement rather than a workaround. Beyond that, implicit-feedback signals (skips, replays) to make the recommender adaptive, and a content-based scorer that uses genre embeddings rather than the categorical genre tags, which is currently powering the sparse-user fallback.

## Tech Stack

Python; LensKit for the ALS implementation; Spotipy for the Spotify Web API; the Last.fm API for the similarity graph; SQLite for artist-metadata caching; Streamlit for the web UI; UMAP and Plotly for the interactive embedding visualizations; deployed via Streamlit Cloud.
