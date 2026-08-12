from __future__ import annotations

import numpy as np
import pandas as pd

from recommender.models import ContentRecommender


def test_tfidf_dims(synthetic_data):
    model = ContentRecommender().fit(synthetic_data["ratings"], items=synthetic_data["items"])
    n_items = len(synthetic_data["items"])
    assert model._item_features is not None
    assert model._item_features.shape[0] == n_items
    assert model.n_features > 0


def test_similar_items_rank_close():
    """An Action-only user should see Action movies in their top recs."""
    items = pd.DataFrame({
        "item_id": [1, 2, 3, 4, 5],
        "title": ["A1", "A2", "Drama1", "Drama2", "Comedy1"],
        "genres": [["Action"], ["Action"], ["Drama"], ["Drama"], ["Comedy"]],
    })
    ratings = pd.DataFrame({
        "user_id": [1, 1],
        "item_id": [1, 2],
        "rating": [5.0, 5.0],
        "timestamp": [1, 2],
    })
    model = ContentRecommender().fit(ratings, items=items)
    recs = model.recommend(user_id=1, k=3, exclude_seen=False)
    # The two Action films should outrank the Drama/Comedy ones.
    assert recs[0] in (1, 2)
    assert recs[1] in (1, 2)


def test_score_returns_correct_shape(synthetic_data):
    model = ContentRecommender().fit(synthetic_data["ratings"], items=synthetic_data["items"])
    scores = model.score(user_id=1, item_ids=[1, 2, 3, 4])
    assert scores.shape == (4,)
    assert scores.dtype == np.float32
