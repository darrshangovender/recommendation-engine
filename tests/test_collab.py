from __future__ import annotations

from recommender.models import CollabALSRecommender


def test_als_fits_and_has_expected_factor_shape(synthetic_data):
    model = CollabALSRecommender(factors=8, iterations=5).fit(
        synthetic_data["ratings"], items=synthetic_data["items"]
    )
    n_users = synthetic_data["ratings"]["user_id"].nunique()
    n_items = synthetic_data["ratings"]["item_id"].nunique()
    assert model.user_factors.shape == (n_users, 8)
    assert model.item_factors.shape == (n_items, 8)


def test_als_recommends_unseen_items(synthetic_data):
    ratings = synthetic_data["ratings"]
    model = CollabALSRecommender(factors=8, iterations=5).fit(ratings, items=synthetic_data["items"])
    seen = set(ratings[ratings["user_id"] == 1]["item_id"].tolist())
    recs = model.recommend(user_id=1, k=3, exclude_seen=True)
    assert not (set(recs) & seen)


def test_als_cold_user_returns_empty(synthetic_data):
    model = CollabALSRecommender(factors=8, iterations=5).fit(
        synthetic_data["ratings"], items=synthetic_data["items"]
    )
    assert model.recommend(user_id=9999, k=5) == []
