"""Implicit ALS collaborative filtering.

We treat MovieLens ratings as implicit feedback: a rating event is a positive
signal, weighted by the explicit rating value (4 stars = stronger preference
than 1 star, but we never represent "didn't like it" — implicit ALS handles
that via the confidence weighting).
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from recommender.models.base import Recommender

# Silence implicit's OpenBLAS-threading warning unless the user has set it.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import implicit  # noqa: E402  (after env vars)

logger = logging.getLogger(__name__)


class CollabALSRecommender(Recommender):
    name = "collab_als"

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        iterations: int = 15,
        alpha: float = 20.0,
        random_state: int = 42,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state

        self._model: implicit.als.AlternatingLeastSquares | None = None
        self._user_id_to_idx: dict[int, int] = {}
        self._item_id_to_idx: dict[int, int] = {}
        self._idx_to_item_id: np.ndarray = np.empty(0, dtype=np.int64)
        self._user_items: csr_matrix | None = None

    def fit(self, ratings: pd.DataFrame, items: pd.DataFrame | None = None) -> "CollabALSRecommender":
        unique_users = np.sort(ratings["user_id"].unique())
        unique_items = np.sort(ratings["item_id"].unique())
        self._user_id_to_idx = {int(u): i for i, u in enumerate(unique_users)}
        self._item_id_to_idx = {int(i): j for j, i in enumerate(unique_items)}
        self._idx_to_item_id = unique_items.astype(np.int64)

        rows = ratings["user_id"].map(self._user_id_to_idx).to_numpy(dtype=np.int32)
        cols = ratings["item_id"].map(self._item_id_to_idx).to_numpy(dtype=np.int32)
        # Confidence = 1 + alpha * rating (Hu/Koren/Volinsky 2008 formulation).
        data = (ratings["rating"].to_numpy(dtype=np.float32))

        self._user_items = csr_matrix(
            (data, (rows, cols)),
            shape=(len(unique_users), len(unique_items)),
            dtype=np.float32,
        )

        self._model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            alpha=self.alpha,
            random_state=self.random_state,
            use_gpu=False,
        )
        logger.info(
            "Training ALS: %d users x %d items, %d events",
            self._user_items.shape[0], self._user_items.shape[1], self._user_items.nnz,
        )
        self._model.fit(self._user_items, show_progress=False)
        return self

    def _user_idx(self, user_id: int) -> int | None:
        return self._user_id_to_idx.get(int(user_id))

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> list[int]:
        if self._model is None or self._user_items is None:
            raise RuntimeError("CollabALSRecommender.fit() must be called before recommend()")
        uidx = self._user_idx(user_id)
        if uidx is None:
            return []  # cold-start user — caller (hybrid) handles fallback
        ids, _scores = self._model.recommend(
            uidx,
            self._user_items[uidx],
            N=k,
            filter_already_liked_items=exclude_seen,
        )
        return [int(self._idx_to_item_id[i]) for i in ids]

    def score(self, user_id: int, item_ids: Iterable[int]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("CollabALSRecommender.fit() must be called before score()")
        uidx = self._user_idx(user_id)
        item_ids = list(item_ids)
        if uidx is None:
            return np.zeros(len(item_ids), dtype=np.float32)
        user_vec = self._model.user_factors[uidx]
        out = np.zeros(len(item_ids), dtype=np.float32)
        for i, iid in enumerate(item_ids):
            iidx = self._item_id_to_idx.get(int(iid))
            if iidx is not None:
                out[i] = float(user_vec @ self._model.item_factors[iidx])
        return out

    @property
    def user_factors(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        return np.asarray(self._model.user_factors)

    @property
    def item_factors(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        return np.asarray(self._model.item_factors)
