"""Popularity baseline.

Recommend the globally most-rated items. Strong baseline that any "real"
recommender should clear comfortably — and one that's surprisingly hard
to beat on novelty-insensitive metrics like raw recall.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from recommender.models.base import Recommender


class PopularityRecommender(Recommender):
    name = "popularity"

    def __init__(self) -> None:
        self._ranked_items: np.ndarray = np.empty(0, dtype=np.int64)
        self._popularity: dict[int, float] = {}
        self._user_seen: dict[int, set[int]] = {}

    def fit(self, ratings: pd.DataFrame, items: pd.DataFrame | None = None) -> "PopularityRecommender":
        counts = ratings.groupby("item_id").size().sort_values(ascending=False)
        self._ranked_items = counts.index.to_numpy(dtype=np.int64)
        # Normalised popularity in [0, 1] for use as a score.
        max_count = int(counts.iloc[0]) if len(counts) else 1
        self._popularity = {int(k): float(v) / max_count for k, v in counts.items()}
        self._user_seen = (
            ratings.groupby("user_id")["item_id"].apply(lambda s: set(s.tolist())).to_dict()
        )
        return self

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> list[int]:
        if self._ranked_items.size == 0:
            raise RuntimeError("PopularityRecommender.fit() must be called before recommend()")
        seen = self._user_seen.get(user_id, set()) if exclude_seen else set()
        out: list[int] = []
        for item in self._ranked_items:
            if len(out) >= k:
                break
            i = int(item)
            if i in seen:
                continue
            out.append(i)
        return out

    def score(self, user_id: int, item_ids: Iterable[int]) -> np.ndarray:
        return np.array([self._popularity.get(int(i), 0.0) for i in item_ids], dtype=np.float32)
