"""Loader + time-split tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommender.data.loader import GENRES, time_split
from tests.conftest import FAKE_N_ITEMS, FAKE_N_RATINGS, FAKE_N_USERS


def test_movielens_shape(movielens):
    """Parser output shape, against the synthetic archive built in conftest."""
    assert len(movielens.ratings) == FAKE_N_RATINGS
    assert set(movielens.ratings.columns) == {"user_id", "item_id", "rating", "timestamp"}
    assert movielens.n_users == FAKE_N_USERS
    assert movielens.n_items == FAKE_N_ITEMS


def test_movielens_dtypes_are_compact(movielens):
    # The loader declares narrow dtypes; a silent widening to object/float64
    # would balloon memory on the real 100K set.
    assert movielens.ratings["user_id"].dtype == np.int32
    assert movielens.ratings["item_id"].dtype == np.int32
    assert movielens.ratings["rating"].dtype == np.float32
    assert movielens.ratings["timestamp"].dtype == np.int64


def test_movielens_no_nan_ratings(movielens):
    assert movielens.ratings["rating"].notna().all()
    assert movielens.ratings["rating"].min() >= 1.0
    assert movielens.ratings["rating"].max() <= 5.0


def test_items_have_genres(movielens):
    assert "genres" in movielens.items.columns
    # Not every item has at least one genre tagged in MovieLens (some are "unknown"),
    # but the column must be a list.
    assert movielens.items["genres"].apply(lambda x: isinstance(x, list)).all()


def test_genre_one_hot_columns_become_a_name_list(movielens):
    """The one-hot → list conversion is the loader's only real transformation."""
    items = movielens.items.set_index("item_id")
    for item_id, row in items.iterrows():
        expected = [g for g in GENRES if row[g] == 1]
        assert row["genres"] == expected, f"item {item_id}"
    # The fixture deliberately includes an untagged item and multi-genre items.
    assert items.loc[1, "genres"] == []
    assert any(len(g) > 1 for g in items["genres"])


def test_item_titles_decode_as_latin1(movielens):
    """u.item is latin-1, not UTF-8 — decoding it wrongly mangles titles."""
    title = movielens.items.set_index("item_id").loc[2, "title"]
    assert "Café" in title


def test_users_table_is_parsed(movielens):
    assert set(movielens.users.columns) == {"user_id", "age", "gender", "occupation", "zip"}
    assert len(movielens.users) == FAKE_N_USERS


def test_loader_does_not_redownload_when_cache_is_populated(fake_movielens_cache, monkeypatch):
    """The cache short-circuit is what keeps the suite offline — pin it.

    Any HTTP call here is a bug: `requests.get` is replaced with a bomb.
    """
    from recommender.data import loader

    def explode(*_a, **_kw):
        raise AssertionError("loader attempted a network request despite a warm cache")

    monkeypatch.setattr(loader.requests, "get", explode)
    data = loader.load_movielens_100k(cache_dir=fake_movielens_cache)
    assert len(data.ratings) == FAKE_N_RATINGS


@pytest.mark.network
def test_real_movielens_100k_shape(movielens_live):
    """The genuine dataset's identity. Opt-in: RECOMMENDER_NETWORK_TESTS=1.

    Kept so the upstream-format assertions above are anchored to reality, but
    never run in CI — a 5MB download per matrix job is not a unit test.
    """
    assert len(movielens_live.ratings) == 100_000
    assert movielens_live.n_users == 943
    assert movielens_live.n_items == 1682


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
