"""Loader + time-split tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommender.data.loader import time_split


def test_movielens_shape(movielens):
    assert len(movielens.ratings) == 100_000
    assert set(movielens.ratings.columns) == {"user_id", "item_id", "rating", "timestamp"}
    assert movielens.n_users == 943
    assert movielens.n_items == 1682


def test_movielens_no_nan_ratings(movielens):
    assert movielens.ratings["rating"].notna().all()
    assert movielens.ratings["rating"].min() >= 1.0
    assert movielens.ratings["rating"].max() <= 5.0


def test_items_have_genres(movielens):
    assert "genres" in movielens.items.columns
    # Not every item has at least one genre tagged in MovieLens (some are "unknown"),
    # but the column must be a list.
    assert movielens.items["genres"].apply(lambda x: isinstance(x, list)).all()


def test_time_split_respects_order():
    ratings = pd.DataFrame({
        "user_id": [1] * 10,
        "item_id": list(range(1, 11)),
        "rating": [3.0] * 10,
        "timestamp": list(range(100, 110)),
    })
    train, test = time_split(ratings, test_frac=0.3)
    assert len(train) + len(test) == 10
    assert train["timestamp"].max() <= test["timestamp"].min()


def test_time_split_validates_frac():
    ratings = pd.DataFrame({"user_id": [1], "item_id": [1], "rating": [3.0], "timestamp": [1]})
    with pytest.raises(ValueError):
        time_split(ratings, test_frac=0.0)
    with pytest.raises(ValueError):
        time_split(ratings, test_frac=1.0)
