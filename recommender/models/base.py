"""Base Recommender interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import pandas as pd


class Recommender(ABC):
    """Abstract interface every model implements.

    Convention: user/item ids are the original 1-indexed MovieLens ids.
    Models map them to internal 0-indexed positions as needed.
    """

    name: str = "abstract"

    @abstractmethod
    def fit(self, ratings: pd.DataFrame, items: pd.DataFrame | None = None) -> "Recommender":
        """Train on a ratings dataframe (columns: user_id, item_id, rating, timestamp)."""

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> list[int]:
        """Return top-``k`` item ids for ``user_id``."""

    def recommend_batch(
        self,
        user_ids: Iterable[int],
        k: int = 10,
        exclude_seen: bool = True,
    ) -> dict[int, list[int]]:
        """Default batch impl — override for speed if needed."""
        return {u: self.recommend(u, k=k, exclude_seen=exclude_seen) for u in user_ids}

    def score(self, user_id: int, item_ids: Iterable[int]) -> np.ndarray:
        """Score a set of items for a user. Optional — used by the hybrid blender.

        Default raises so models that don't support arbitrary scoring fail loudly
        rather than silently returning zeros.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement score()")
