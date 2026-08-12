"""Offline ranking metrics: Recall@K, Precision@K, NDCG@K, coverage, novelty."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from recommender.models.base import Recommender


@dataclass
class EvalResult:
    model: str
    k: int
    recall: float
    precision: float
    ndcg: float
    coverage: float       # fraction of catalogue surfaced across all users
    novelty: float        # mean (1 - normalised popularity) of recommended items
    n_users_evaluated: int

    def as_row(self) -> dict[str, float | str | int]:
        return {
            "model": self.model,
            "k": self.k,
            "recall@k": round(self.recall, 4),
            "precision@k": round(self.precision, 4),
            "ndcg@k": round(self.ndcg, 4),
            "coverage": round(self.coverage, 4),
            "novelty": round(self.novelty, 4),
            "n_users": self.n_users_evaluated,
        }


def _recall_at_k(recs: list[int], truth: set[int], k: int) -> float:
    if not truth:
        return 0.0
    hits = sum(1 for r in recs[:k] if r in truth)
    return hits / min(len(truth), k)


def _precision_at_k(recs: list[int], truth: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for r in recs[:k] if r in truth)
    return hits / k


def _ndcg_at_k(recs: list[int], truth: set[int], k: int) -> float:
    if not truth:
        return 0.0
    dcg = 0.0
    for i, r in enumerate(recs[:k]):
        if r in truth:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(
    model: Recommender,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
    catalog_size: int | None = None,
    user_sample: int | None = None,
    seed: int = 42,
) -> EvalResult:
    """Evaluate ``model`` on a leave-future-out test set.

    Only users present in BOTH train and test are scored — a model can't
    be blamed for missing users it never saw, and we need ground-truth
    purchases in the test window to compute recall.
    """
    train_users = set(train["user_id"].unique())
    test_users = set(test["user_id"].unique())
    eval_users = sorted(train_users & test_users)

    if user_sample is not None and len(eval_users) > user_sample:
        rng = np.random.default_rng(seed)
        eval_users = sorted(rng.choice(eval_users, size=user_sample, replace=False).tolist())

    truth_by_user: dict[int, set[int]] = (
        test.groupby("user_id")["item_id"].apply(lambda s: set(int(i) for i in s)).to_dict()
    )

    # Item popularity from TRAIN — never peek at test.
    item_counts = train.groupby("item_id").size()
    max_count = int(item_counts.max()) if len(item_counts) else 1
    norm_pop = (item_counts / max_count).to_dict()

    if catalog_size is None:
        catalog_size = int(train["item_id"].nunique())

    recalls: list[float] = []
    precisions: list[float] = []
    ndcgs: list[float] = []
    all_recs: set[int] = set()
    novelty_scores: list[float] = []

    for uid in eval_users:
        recs = model.recommend(uid, k=k, exclude_seen=True)
        truth = truth_by_user.get(uid, set())
        recalls.append(_recall_at_k(recs, truth, k))
        precisions.append(_precision_at_k(recs, truth, k))
        ndcgs.append(_ndcg_at_k(recs, truth, k))
        all_recs.update(recs)
        for r in recs:
            novelty_scores.append(1.0 - float(norm_pop.get(int(r), 0.0)))

    coverage = len(all_recs) / catalog_size if catalog_size else 0.0
    novelty = float(np.mean(novelty_scores)) if novelty_scores else 0.0

    return EvalResult(
        model=model.name,
        k=k,
        recall=float(np.mean(recalls)) if recalls else 0.0,
        precision=float(np.mean(precisions)) if precisions else 0.0,
        ndcg=float(np.mean(ndcgs)) if ndcgs else 0.0,
        coverage=coverage,
        novelty=novelty,
        n_users_evaluated=len(eval_users),
    )


def results_table(results: Iterable[EvalResult]) -> pd.DataFrame:
    return pd.DataFrame([r.as_row() for r in results])
