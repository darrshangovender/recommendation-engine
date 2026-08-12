"""MovieLens-100K loader.

Downloads the dataset on first run, caches it under ``~/.recommender_cache/``.
The 100K dataset is ~5MB compressed and ships under a research license that
permits redistribution and modification — see the ``README`` inside the archive.
"""

from __future__ import annotations

import io
import logging
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DEFAULT_CACHE_DIR = Path(os.environ.get("RECOMMENDER_CACHE", Path.home() / ".recommender_cache"))

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


@dataclass
class MovieLensData:
    """Bundle of the loaded MovieLens-100K artefacts."""

    ratings: pd.DataFrame   # columns: user_id, item_id, rating, timestamp
    items: pd.DataFrame     # columns: item_id, title, release_date, genres (str list), + one-hot genre cols
    users: pd.DataFrame     # columns: user_id, age, gender, occupation, zip

    @property
    def n_users(self) -> int:
        return int(self.ratings["user_id"].max())

    @property
    def n_items(self) -> int:
        return int(self.ratings["item_id"].max())


def _download_and_extract(cache_dir: Path) -> Path:
    """Download the ml-100k zip into ``cache_dir`` and extract it. Idempotent."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = cache_dir / "ml-100k"
    if extract_dir.exists() and (extract_dir / "u.data").exists():
        return extract_dir

    logger.info("Downloading MovieLens-100K from %s ...", MOVIELENS_100K_URL)
    resp = requests.get(MOVIELENS_100K_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(cache_dir)
    if not (extract_dir / "u.data").exists():
        raise RuntimeError(f"Expected u.data inside {extract_dir} after extraction")
    return extract_dir


def load_movielens_100k(cache_dir: Path | None = None) -> MovieLensData:
    """Load MovieLens-100K, downloading + caching on first run."""
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    data_dir = _download_and_extract(cache_dir)

    ratings = pd.read_csv(
        data_dir / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": np.int32, "item_id": np.int32, "rating": np.float32, "timestamp": np.int64},
    )

    item_cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRES
    items = pd.read_csv(
        data_dir / "u.item",
        sep="|",
        names=item_cols,
        encoding="latin-1",
        dtype={"item_id": np.int32},
    )
    # Build a list-of-genres column for downstream feature builders.
    items["genres"] = items[GENRES].apply(
        lambda row: [g for g, on in zip(GENRES, row.values) if on == 1], axis=1
    )

    users = pd.read_csv(
        data_dir / "u.user",
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip"],
        dtype={"user_id": np.int32, "age": np.int32},
    )

    # Sanity: no NaN ratings, ids start at 1.
    assert ratings["rating"].notna().all(), "Ratings contain NaN"
    assert ratings["user_id"].min() >= 1 and ratings["item_id"].min() >= 1

    return MovieLensData(ratings=ratings, items=items, users=users)


def time_split(
    ratings: pd.DataFrame, test_frac: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split.

    The cutoff is the timestamp quantile at ``1 - test_frac``; events at or
    before the cutoff go into train, later events into test. This preserves
    time order — train cannot peek into the future — which matters for
    honest recall evaluation.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    cutoff = ratings["timestamp"].quantile(1.0 - test_frac)
    train = ratings[ratings["timestamp"] <= cutoff].copy()
    test = ratings[ratings["timestamp"] > cutoff].copy()
    return train, test
