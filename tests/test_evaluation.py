"""Contrived recall/precision/NDCG cases — verifies the metric math."""

from __future__ import annotations

import math

import pandas as pd

from recommender.evaluation import (
    _ndcg_at_k,
    _precision_at_k,
    _recall_at_k,
    evaluate,
)
from recommender.models.base import Recommender


def test_recall_at_k_perfect():
    assert _recall_at_k([1, 2, 3], truth={1, 2, 3}, k=3) == 1.0


def test_recall_at_k_partial():
    # 1 of 2 truth items recovered in top-3 → recall = 0.5
    assert _recall_at_k([1, 4, 5], truth={1, 2}, k=3) == 0.5


def test_recall_denominator_is_full_truth_set_not_capped_at_k():
    """The denominator must be |relevant|, not min(|relevant|, k).

    The capped form inflates any user with more relevant items than slots: 5
    hits against 20 held-out items at k=10 is 0.25 recall, but capping the
    denominator at k reports 0.50. Every case where |truth| <= k agrees, which
    is why the rest of this file never caught it.
    """
    truth = set(range(1, 21))          # 20 relevant items
    recs = [1, 2, 3, 4, 5, 90, 91, 92, 93, 94]   # 5 hits in the top 10
    assert _recall_at_k(recs, truth=truth, k=10) == 0.25


def test_recall_agrees_with_capped_form_when_truth_fits_in_k():
    """Sanity check that the fix is a no-op on the common case."""
    assert _recall_at_k([1, 2, 9], truth={1, 2}, k=10) == 1.0


def test_precision_at_k():
    assert _precision_at_k([1, 2, 9], truth={1, 2}, k=3) == 2 / 3


def test_ndcg_at_k_known_value():
    # Hits at positions 1 and 3 (1-indexed). DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    # IDCG (2 hits) = 1 + 1/log2(3) ≈ 1.6309
    recs = [1, 99, 2]
    ndcg = _ndcg_at_k(recs, truth={1, 2}, k=3)
    expected = (1.0 + 0.5) / (1.0 + 1.0 / math.log2(3))
    assert abs(ndcg - expected) < 1e-9


def test_evaluate_end_to_end():
    """A trivial model that always returns [1, 2] should hit recall=1 on a truth of {1, 2}."""

    class FixedModel(Recommender):
        name = "fixed"
        def fit(self, ratings, items=None):
            return self
        def recommend(self, user_id, k=10, exclude_seen=True):
            return [1, 2, 3]

    train = pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "item_id": [5, 6, 5, 6],
        "rating": [5.0, 5.0, 5.0, 5.0],
        "timestamp": [1, 2, 3, 4],
    })
    test = pd.DataFrame({
        "user_id": [1, 1, 2],
        "item_id": [1, 2, 1],
        "rating": [5.0, 5.0, 5.0],
        "timestamp": [10, 11, 12],
    })
    model = FixedModel().fit(train)
    res = evaluate(model, train, test, k=3)
    # User 1: truth={1,2}, recs[:3] hit both → recall=1.0
    # User 2: truth={1},   recs[:3] hit 1   → recall=1.0
    assert abs(res.recall - 1.0) < 1e-9
    assert res.n_users_evaluated == 2
