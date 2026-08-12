"""Shared test fixtures.

Tests use a small synthetic dataset by default — fast and hermetic. The
``movielens`` fixture (marked ``slow``) hits the network on first run; subsequent
runs read from the local cache.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_data() -> dict[str, pd.DataFrame]:
    """A tiny dataset: 6 users, 8 items, 3 genres, ~30 events."""
    rng = np.random.default_rng(0)
    items = pd.DataFrame({
        "item_id": list(range(1, 9)),
        "title": [f"Movie {i}" for i in range(1, 9)],
        "genres": [
            ["Action"], ["Action", "Sci-Fi"], ["Drama"], ["Drama", "Romance"],
            ["Comedy"], ["Comedy", "Romance"], ["Action", "Drama"], ["Sci-Fi"],
        ],
    })
    users = list(range(1, 7))
    rows = []
    base_ts = 1_000_000
    for u in users:
        # Each user rates 4-6 items.
        n = rng.integers(4, 7)
        chosen = rng.choice(items["item_id"], size=n, replace=False)
        for i, iid in enumerate(chosen):
            rows.append({
                "user_id": int(u),
                "item_id": int(iid),
                "rating": float(rng.integers(1, 6)),
                "timestamp": base_ts + u * 1000 + i,
            })
    ratings = pd.DataFrame(rows).astype({"user_id": np.int32, "item_id": np.int32, "rating": np.float32, "timestamp": np.int64})
    return {"ratings": ratings, "items": items}


@pytest.fixture(scope="session")
def movielens():
    """Real MovieLens-100K (cached). Skipped if download fails (offline env)."""
    from recommender.data.loader import load_movielens_100k
    try:
        return load_movielens_100k()
    except Exception as exc:  # network down etc.
        pytest.skip(f"MovieLens not available: {exc}")
