from __future__ import annotations

from recommender.models import PopularityRecommender


def test_topn_no_duplicates(synthetic_data):
    model = PopularityRecommender().fit(synthetic_data["ratings"], items=synthetic_data["items"])
    recs = model.recommend(user_id=1, k=5, exclude_seen=False)
    assert len(recs) == len(set(recs))


def test_topn_stable(synthetic_data):
    model = PopularityRecommender().fit(synthetic_data["ratings"], items=synthetic_data["items"])
    a = model.recommend(user_id=1, k=3, exclude_seen=False)
    b = model.recommend(user_id=1, k=3, exclude_seen=False)
    assert a == b


def test_excludes_seen(synthetic_data):
    ratings = synthetic_data["ratings"]
    model = PopularityRecommender().fit(ratings, items=synthetic_data["items"])
    seen = set(ratings[ratings["user_id"] == 1]["item_id"].tolist())
    recs = model.recommend(user_id=1, k=8, exclude_seen=True)
    assert not (set(recs) & seen)


def test_score_is_in_unit_interval(synthetic_data):
    model = PopularityRecommender().fit(synthetic_data["ratings"], items=synthetic_data["items"])
    scores = model.score(user_id=1, item_ids=[1, 2, 3])
    assert ((scores >= 0.0) & (scores <= 1.0)).all()
