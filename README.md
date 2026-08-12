# E-Commerce Recommendation Engine

> Hybrid recommender for an e-commerce client. Embedding-driven content similarity for cold-start products, behavioural signals for warm users. Drove measurable lift in average order value and subscription conversion.

**Stack:** Python · scikit-learn · sentence-transformers · PostgreSQL · FastAPI · Redis (cache)

---

## What it does

Two recommender heads, blended at serve time:

1. **Content-based** — product descriptions are embedded with a sentence-transformer; cosine similarity surfaces visually and semantically related items. Works on day one for new SKUs (cold start).
2. **Collaborative** — user×product co-purchase matrix; matrix-factorisation (implicit ALS) for users with sufficient history.

A blender weights the two heads per request based on user history depth — pure content for new users, pure collaborative for power users, a smooth mix in between.

## Why hybrid

Cold-start is the practical killer for e-commerce recommenders. Every retailer has a long tail of new SKUs and a long tail of new users. A pure collaborative model is dead silent on both. A pure content model is OK at cold-start but loses the "people who bought X also bought Y" signal that really drives basket size.

The hybrid is boring engineering, but it's the design that wins in production.

## Architecture

```
                       ┌────────────┐
   Product catalogue ─▶│ Embedder   │─▶ pgvector (item × emb)
                       └────────────┘
                              │
                              ▼
   User events ────▶ ALS trainer ─▶ user × item factors (Postgres)
                              │
   Request ──▶ FastAPI ──▶ ┌──Blender──┐
                           │ history?  │
                           │ < N events: content-only
                           │ ≥ N events: 0.7 collab + 0.3 content
                           └─────┬─────┘
                                 ▼
                            Redis cache
                                 ▼
                              Response
```

## Repo structure

```
.
├── pipeline/
│   ├── embed_catalog.py   # sentence-transformers
│   ├── train_als.py       # implicit ALS on co-purchase matrix
│   └── refresh.py         # nightly refresh
├── api/
│   ├── main.py
│   ├── content.py
│   ├── collab.py
│   └── blender.py
├── eval/
│   └── recall_at_k.py
└── notebooks/
    └── exploration.ipynb
```

## Why this trained well

Three things mattered more than the algorithm choice:

1. **Implicit feedback weighting.** Co-purchases weighted higher than co-views, which were weighted higher than co-clicks. Standard but easy to skip.
2. **Time decay.** Events older than 6 months get exponentially down-weighted — fashion and seasonality matter.
3. **Negative sampling.** Random unseen items as negatives, with a frequency-based correction so we don't oversample rare items.

## Production results

- **Measurable lift** in average order value on the recommendation slot (numbers under NDA).
- Conversion uplift on the subscription product line vs the previous rule-based "frequently bought together" widget.
- Sub-100ms p95 serve latency for 95th-percentile users (Redis-cached top-K).

## Eval

Offline `recall@k` on a time-split holdout — train on events through month T, evaluate on month T+1 purchases.

```bash
uv run python eval/recall_at_k.py --k 10 --split 2025-09-01
# → recall@10: 0.41  (baseline rule-based: 0.18)
```

## Local setup

```bash
docker compose up -d
uv sync
uv run python pipeline/embed_catalog.py --catalog ./sample_catalog.csv
uv run python pipeline/train_als.py
uv run uvicorn api.main:app --reload
```

## Author

Darrshan Govender · Founder, [Agulhas Code](https://agulhascode.co.za)
