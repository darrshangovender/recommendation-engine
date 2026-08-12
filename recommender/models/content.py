"""Content-based recommender — TF-IDF on item genres + title tokens, cosine similarity.

User profile = mean of TF-IDF vectors of positively rated items (rating >= 4),
falling back to all rated items if the user has no high ratings.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from recommender.models.base import Recommender


def _build_item_text(items: pd.DataFrame) -> pd.Series:
    """Concatenate genres + title tokens into a bag-of-words string per item."""
    def row_text(row: pd.Series) -> str:
        genres = " ".join(row["genres"]) if isinstance(row["genres"], list) else ""
        title = str(row.get("title", "")).lower()
        # Strip the trailing "(year)" — it adds noise to the TF-IDF.
        if "(" in title:
            title = title[: title.rfind("(")].strip()
        return f"{genres} {title}".strip()

    return items.apply(row_text, axis=1)


class ContentRecommender(Recommender):
    name = "content"

    def __init__(self, min_positive_rating: float = 4.0) -> None:
        self.min_positive_rating = float(min_positive_rating)
        self._vectorizer: TfidfVectorizer | None = None
        self._item_features: csr_matrix | None = None  # normalised rows
        self._item_id_to_idx: dict[int, int] = {}
        self._idx_to_item_id: np.ndarray = np.empty(0, dtype=np.int64)
        self._user_profiles: dict[int, csr_matrix] = {}
        self._user_seen: dict[int, set[int]] = {}

    def fit(self, ratings: pd.DataFrame, items: pd.DataFrame | None = None) -> "ContentRecommender":
        if items is None:
            raise ValueError("ContentRecommender requires the items dataframe")

        items = items.sort_values("item_id").reset_index(drop=True)
        self._idx_to_item_id = items["item_id"].to_numpy(dtype=np.int64)
        self._item_id_to_idx = {int(iid): i for i, iid in enumerate(self._idx_to_item_id)}

        text = _build_item_text(items)
        self._vectorizer = TfidfVectorizer(token_pattern=r"[A-Za-z][A-Za-z\-]+")
        item_features = self._vectorizer.fit_transform(text)
        self._item_features = normalize(item_features, norm="l2", axis=1)

        self._user_seen = (
            ratings.groupby("user_id")["item_id"].apply(lambda s: set(s.tolist())).to_dict()
        )
        self._build_user_profiles(ratings)
        return self

    def _build_user_profiles(self, ratings: pd.DataFrame) -> None:
        assert self._item_features is not None
        for uid, group in ratings.groupby("user_id"):
            pos = group[group["rating"] >= self.min_positive_rating]
            if pos.empty:
                pos = group  # fallback so every rater gets a profile
            idxs = [self._item_id_to_idx[int(i)] for i in pos["item_id"] if int(i) in self._item_id_to_idx]
            if not idxs:
                continue
            stacked = vstack([self._item_features[i] for i in idxs])
            profile = csr_matrix(stacked.mean(axis=0))
            self._user_profiles[int(uid)] = normalize(profile, norm="l2", axis=1)

    def _score_all_items(self, user_id: int) -> np.ndarray:
        assert self._item_features is not None
        profile = self._user_profiles.get(user_id)
        if profile is None:
            return np.zeros(self._item_features.shape[0], dtype=np.float32)
        scores = (self._item_features @ profile.T).toarray().ravel()
        return scores.astype(np.float32)

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> list[int]:
        if self._item_features is None:
            raise RuntimeError("ContentRecommender.fit() must be called before recommend()")
        scores = self._score_all_items(user_id)
        if exclude_seen:
            for iid in self._user_seen.get(user_id, set()):
                idx = self._item_id_to_idx.get(int(iid))
                if idx is not None:
                    scores[idx] = -np.inf
        if not np.isfinite(scores).any():
            return []
        top_idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [int(self._idx_to_item_id[i]) for i in top_idx if np.isfinite(scores[i])]

    def score(self, user_id: int, item_ids: Iterable[int]) -> np.ndarray:
        scores = self._score_all_items(user_id)
        out = np.zeros(len(list(item_ids := list(item_ids))), dtype=np.float32)
        for i, iid in enumerate(item_ids):
            idx = self._item_id_to_idx.get(int(iid))
            if idx is not None:
                out[i] = scores[idx]
        return out

    @property
    def n_features(self) -> int:
        return 0 if self._item_features is None else self._item_features.shape[1]
