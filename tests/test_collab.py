from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

from recommender.models import CollabALSRecommender, collab_als


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


# --- Backend selection -----------------------------------------------------
#
# `implicit` ships compiled extensions and is not installable everywhere, so
# the ALS model must stay usable without it. These tests pin that contract.

def test_importing_models_does_not_require_implicit():
    """A top-level `import implicit` here previously made the whole package
    uncollectable wherever the wheel was unavailable."""
    code = "import sys; import recommender.models; print(int('implicit' in sys.modules))"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == "0", "importing recommender.models pulled in implicit"


def test_auto_backend_resolves_to_numpy_when_implicit_is_absent(monkeypatch):
    monkeypatch.setattr(collab_als, "implicit_available", lambda: False)
    assert CollabALSRecommender(backend="auto").resolved_backend == "numpy"


def test_auto_backend_prefers_implicit_when_present(monkeypatch):
    monkeypatch.setattr(collab_als, "implicit_available", lambda: True)
    assert CollabALSRecommender(backend="auto").resolved_backend == "implicit"


def test_explicit_backend_is_honoured():
    assert CollabALSRecommender(backend="numpy").resolved_backend == "numpy"


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="backend must be one of"):
        CollabALSRecommender(backend="tensorflow")


def test_requesting_implicit_when_missing_fails_loudly(monkeypatch, synthetic_data):
    """Better a clear install hint than a silent switch to the slow solver."""
    monkeypatch.setattr(collab_als, "implicit_available", lambda: False)
    model = CollabALSRecommender(factors=4, iterations=2, backend="implicit")
    with pytest.raises(RuntimeError, match=r"pip install recommender\[als\]"):
        model.fit(synthetic_data["ratings"])


# --- Numpy solver quality --------------------------------------------------

def _two_cluster_ratings(drop=None):
    """Users 1-10 like items 1-5; users 11-20 like items 6-10."""
    rows = []
    for user in range(1, 11):
        for item in range(1, 6):
            rows.append((user, item, 5.0, user * 10 + item))
    for user in range(11, 21):
        for item in range(6, 11):
            rows.append((user, item, 5.0, user * 10 + item))
    if drop is not None:
        rows = [r for r in rows if (r[0], r[1]) != drop]
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def test_numpy_als_separates_taste_clusters():
    model = CollabALSRecommender(factors=8, iterations=20, backend="numpy")
    model.fit(_two_cluster_ratings())
    in_cluster = model.score(1, range(1, 6))
    out_cluster = model.score(1, range(6, 11))
    assert in_cluster.mean() > out_cluster.mean() + 0.5


def test_numpy_als_recovers_a_held_out_preference():
    """The real test of a recommender: rank an item the user liked but never
    saw during training above everything else it hasn't seen."""
    model = CollabALSRecommender(factors=8, iterations=20, backend="numpy")
    model.fit(_two_cluster_ratings(drop=(1, 3)))
    assert model.recommend(user_id=1, k=3)[0] == 3


def test_numpy_als_is_deterministic():
    ratings = _two_cluster_ratings()
    first = CollabALSRecommender(factors=8, iterations=10, backend="numpy").fit(ratings)
    second = CollabALSRecommender(factors=8, iterations=10, backend="numpy").fit(ratings)
    assert first.recommend(1, k=5) == second.recommend(1, k=5)


def test_score_returns_zeros_for_cold_user():
    model = CollabALSRecommender(factors=4, iterations=5, backend="numpy")
    model.fit(_two_cluster_ratings())
    assert not model.score(9999, [1, 2, 3]).any()


def test_recommend_before_fit_raises():
    with pytest.raises(RuntimeError, match="must be called before recommend"):
        CollabALSRecommender(backend="numpy").recommend(user_id=1)


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError, match="must be called before score"):
        CollabALSRecommender(backend="numpy").score(user_id=1, item_ids=[1])


def test_exclude_seen_holds_when_user_has_seen_almost_everything():
    """Regression: `implicit` pads its top-N with items it had already masked,
    so a user who has seen most of the catalogue got seen items back."""
    rows = [(1, i, 5.0, 100 + i) for i in range(1, 9)]  # user 1 saw 8 of 10
    rows += [(2, i, 4.0, 200 + i) for i in range(1, 11)]  # user 2 gives items 9,10 a factor
    ratings = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])

    model = CollabALSRecommender(factors=4, iterations=5).fit(ratings)
    seen = set(range(1, 9))
    recs = model.recommend(user_id=1, k=10, exclude_seen=True)

    assert not (set(recs) & seen), f"seen items leaked into recommendations: {recs}"
    # Only two unseen items exist, so a shorter list is the right answer.
    assert len(recs) <= 2


def test_exclude_seen_false_may_return_seen_items():
    rows = [(1, i, 5.0, 100 + i) for i in range(1, 9)]
    rows += [(2, i, 4.0, 200 + i) for i in range(1, 11)]
    ratings = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    model = CollabALSRecommender(factors=4, iterations=5).fit(ratings)
    assert len(model.recommend(user_id=1, k=10, exclude_seen=False)) == 10
