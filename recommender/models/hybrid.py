"""Hybrid recommender — weighted blend of content + collaborative scores.

The blend weight ``alpha`` is per-user:
    alpha = 1.0  → pure content (cold-start safe)
    alpha = 0.0  → pure collaborative
    alpha = 0.3  → "warm" default — collab leads, content keeps long-tail diversity

We pick alpha from interaction count: users with < ``cold_threshold`` events
get content-heavy weights, ramping linearly down to ``warm_alpha`` once they
hit ``warm_threshold`` events.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from recommender.models.base import Recommender
from recommender.models.collab_als import CollabALSRecommender
from recommender.models.content import ContentRecommender
from recommender.models.popularity import PopularityRecommender


def _minmax(arr: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]. Constant vectors collapse to zeros."""
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


class HybridRecommender(Recommender):
    name = "hybrid"

    def __init__(
        self,
        content: ContentRecommender | None = None,
        collab: CollabALSRecommender | None = None,
        cold_threshold: int = 5,
        warm_threshold: int = 30,
        warm_alpha: float = 0.3,
        candidate_pool: int = 200,
    ) -> None:
        self.content = content or ContentRecommender()
        self.collab = collab or CollabALSRecommender()
        self.cold_threshold = cold_threshold
        self.warm_threshold = warm_threshold
        self.warm_alpha = warm_alpha
        self.candidate_pool = candidate_pool
        # Popularity is used purely as a cold-start fallback for users absent
        # from training entirely (no content profile either).
        self._popularity = PopularityRecommender()
        self._user_event_count: dict[int, int] = {}
        self._all_item_ids: np.ndarray = np.empty(0, dtype=np.int64)
        self._fixed_alpha: float | None = None

    def with_fixed_alpha(self, alpha: float) -> "HybridRecommender":
        """For testing — pin alpha regardless of user history depth."""
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self._fixed_alpha = float(alpha)
        return self

    def fit(self, ratings: pd.DataFrame, items: pd.DataFrame | None = None) -> "HybridRecommender":
        self.content.fit(ratings, items=items)
        self.collab.fit(ratings, items=items)
        self._popularity.fit(ratings, items=items)
        self._user_event_count = ratings.groupby("user_id").size().to_dict()
        self._all_item_ids = np.sort(ratings["item_id"].unique()).astype(np.int64)
        return self

    def _alpha_for(self, user_id: int) -> float:
        if self._fixed_alpha is not None:
            return self._fixed_alpha
        n = self._user_event_count.get(int(user_id), 0)
        if n <= self.cold_threshold:
            return 1.0
        if n >= self.warm_threshold:
            return self.warm_alpha
        # Linear ramp between the two thresholds.
        span = self.warm_threshold - self.cold_threshold
        frac = (n - self.cold_threshold) / span
        return float(1.0 - frac * (1.0 - self.warm_alpha))

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> list[int]:
        if self._all_item_ids.size == 0:
            raise RuntimeError("HybridRecommender.fit() must be called before recommend()")

        alpha = self._alpha_for(user_id)
        # Candidate pool: union of top-N from each head, plus popularity backfill.
        pool_size = max(self.candidate_pool, k * 4)
        candidates: set[int] = set()
        try:
            candidates.update(self.content.recommend(user_id, k=pool_size, exclude_seen=exclude_seen))
        except Exception:
            pass
        try:
            candidates.update(self.collab.recommend(user_id, k=pool_size, exclude_seen=exclude_seen))
        except Exception:
            pass
        if not candidates:
            return self._popularity.recommend(user_id, k=k, exclude_seen=exclude_seen)

        items = np.array(sorted(candidates), dtype=np.int64)
        content_raw = self.content.score(user_id, items)
        try:
            collab_raw = self.collab.score(user_id, items)
        except RuntimeError:
            collab_raw = np.zeros_like(content_raw)
        blended = alpha * _minmax(content_raw) + (1.0 - alpha) * _minmax(collab_raw)

        top_idx = np.argsort(-blended)[:k]
        return [int(items[i]) for i in top_idx]

    def score(self, user_id: int, item_ids: Iterable[int]) -> np.ndarray:
        item_ids = list(item_ids)
        items_arr = np.array(item_ids, dtype=np.int64)
        alpha = self._alpha_for(user_id)
        content_raw = self.content.score(user_id, items_arr)
        try:
            collab_raw = self.collab.score(user_id, items_arr)
        except RuntimeError:
            collab_raw = np.zeros_like(content_raw)
        return alpha * _minmax(content_raw) + (1.0 - alpha) * _minmax(collab_raw)
