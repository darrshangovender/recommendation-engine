"""Implicit ALS collaborative filtering.

We treat MovieLens ratings as implicit feedback: a rating event is a positive
signal, weighted by the explicit rating value (4 stars = stronger preference
than 1 star, but we never represent "didn't like it" — implicit ALS handles
that via the confidence weighting).
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from recommender.models._als_numpy import NumpyALS
from recommender.models.base import Recommender

# Silence implicit's OpenBLAS-threading warning unless the user has set it.
# Must happen before implicit is imported, hence the module-level assignment
# even though the import itself is now deferred.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

logger = logging.getLogger(__name__)

VALID_BACKENDS = ("auto", "implicit", "numpy")


def implicit_available() -> bool:
    """True if the compiled ``implicit`` package can be imported.

    ``implicit`` ships C extensions and has no wheel for every Python/OS
    combination, so its absence is an expected state rather than an error —
    see :mod:`recommender.models._als_numpy`.
    """
    return importlib.util.find_spec("implicit") is not None


class CollabALSRecommender(Recommender):
    name = "collab_als"

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        iterations: int = 15,
        alpha: float = 20.0,
        random_state: int = 42,
        backend: str = "auto",
    ) -> None:
        if backend not in VALID_BACKENDS:
            raise ValueError(f"backend must be one of {VALID_BACKENDS}, got {backend!r}")
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state
        self.backend = backend

        self._model: Any | None = None
        self._user_id_to_idx: dict[int, int] = {}
        self._item_id_to_idx: dict[int, int] = {}
        self._idx_to_item_id: np.ndarray = np.empty(0, dtype=np.int64)
        self._user_items: csr_matrix | None = None

    def fit(self, ratings: pd.DataFrame, items: pd.DataFrame | None = None) -> CollabALSRecommender:
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

        self._model = self._build_model()
        logger.info(
            "Training ALS (%s backend): %d users x %d items, %d events",
            self.resolved_backend,
            self._user_items.shape[0], self._user_items.shape[1], self._user_items.nnz,
        )
        self._model.fit(self._user_items, show_progress=False)
        return self

    @property
    def resolved_backend(self) -> str:
        """Which solver ``backend="auto"`` actually selects on this machine."""
        if self.backend != "auto":
            return self.backend
        return "implicit" if implicit_available() else "numpy"

    def _build_model(self) -> Any:
        """Instantiate the ALS solver for the resolved backend.

        ``implicit`` is imported here rather than at module scope so that
        importing ``recommender.models`` never depends on a compiled wheel —
        that import failure previously took the whole test suite down with it.
        """
        backend = self.resolved_backend
        if backend == "implicit":
            if not implicit_available():
                raise RuntimeError(
                    "backend='implicit' requested but the package is not installed. "
                    "Install it with `pip install recommender[als]`, or use backend='numpy'."
                )
            implicit_als = importlib.import_module("implicit.als")
            return implicit_als.AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
                alpha=self.alpha,
                random_state=self.random_state,
                use_gpu=False,
            )
        return NumpyALS(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            alpha=self.alpha,
            random_state=self.random_state,
        )

    def _user_idx(self, user_id: int) -> int | None:
        return self._user_id_to_idx.get(int(user_id))

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> list[int]:
        if self._model is None or self._user_items is None:
            raise RuntimeError("CollabALSRecommender.fit() must be called before recommend()")
        uidx = self._user_idx(user_id)
        if uidx is None:
            return []  # cold-start user — caller (hybrid) handles fallback
        user_row = self._user_items[uidx]
        ids, _scores = self._model.recommend(
            uidx,
            user_row,
            N=k,
            filter_already_liked_items=exclude_seen,
        )

        # `implicit` pads its top-N up to N even when fewer unseen items exist,
        # backfilling with entries it had already masked to -FLT_MAX. Trusting
        # its output verbatim therefore leaks *seen* items back into the
        # recommendations whenever a user has covered most of the catalogue.
        # We hold the authoritative seen-set, so re-apply the filter here rather
        # than depending on any backend's padding behaviour. Returning fewer
        # than k items is the correct answer when the catalogue is exhausted.
        if exclude_seen:
            seen_idx = set(user_row.indices.tolist())
            return [int(self._idx_to_item_id[i]) for i in ids if int(i) not in seen_idx]
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
