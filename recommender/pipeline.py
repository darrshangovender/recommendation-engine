"""End-to-end pipeline: load → split → train → evaluate → sample recommendations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from recommender.data.loader import MovieLensData, load_movielens_100k, time_split
from recommender.evaluation import EvalResult, evaluate, results_table
from recommender.models import (
    CollabALSRecommender,
    ContentRecommender,
    HybridRecommender,
    PopularityRecommender,
    Recommender,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineOutput:
    results: list[EvalResult]
    sample_recs: dict[str, dict[int, list[int]]]  # model_name -> {user_id -> [item_ids]}
    models: dict[str, Recommender] = field(default_factory=dict)

    def table(self) -> pd.DataFrame:
        return results_table(self.results)


class Pipeline:
    """Train every model, evaluate each, emit sample recommendations.

    Default sample users span three profiles:
        - cold user  (fewest train interactions)
        - active user (most train interactions)
        - niche user (rates items in the smallest-mean-popularity quartile)
    """

    def __init__(
        self,
        k: int = 10,
        test_frac: float = 0.2,
        user_sample: int | None = 300,
        sample_user_ids: Sequence[int] | None = None,
        models: Sequence[Recommender] | None = None,
        seed: int = 42,
    ) -> None:
        self.k = k
        self.test_frac = test_frac
        self.user_sample = user_sample
        self.sample_user_ids = sample_user_ids
        self._models = models
        self.seed = seed

    def _default_models(self) -> list[Recommender]:
        return [
            PopularityRecommender(),
            ContentRecommender(),
            CollabALSRecommender(),
            HybridRecommender(),
        ]

    def _pick_sample_users(self, train: pd.DataFrame) -> list[int]:
        if self.sample_user_ids:
            return [int(u) for u in self.sample_user_ids]
        per_user = train.groupby("user_id").size().sort_values()
        cold = int(per_user.index[0])
        active = int(per_user.index[-1])
        # Niche taste: average popularity of items they rated is in the bottom quartile.
        item_pop = train.groupby("item_id").size()
        user_mean_pop = train.groupby("user_id")["item_id"].apply(
            lambda s: item_pop.reindex(s).mean()
        )
        niche = int(user_mean_pop.sort_values().index[0])
        # De-dup while preserving order.
        seen: set[int] = set()
        out: list[int] = []
        for u in (cold, active, niche):
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out

    def run(self, data: MovieLensData | None = None) -> PipelineOutput:
        if data is None:
            logger.info("Loading MovieLens-100K ...")
            data = load_movielens_100k()
        train, test = time_split(data.ratings, test_frac=self.test_frac)
        logger.info("Train events: %d | Test events: %d", len(train), len(test))

        models = list(self._models) if self._models is not None else self._default_models()
        results: list[EvalResult] = []
        sample_recs: dict[str, dict[int, list[int]]] = {}

        sample_users = self._pick_sample_users(train)

        for m in models:
            logger.info("Training %s ...", m.name)
            m.fit(train, items=data.items)
            logger.info("Evaluating %s ...", m.name)
            res = evaluate(
                m, train, test,
                k=self.k,
                catalog_size=int(data.items["item_id"].nunique()),
                user_sample=self.user_sample,
                seed=self.seed,
            )
            results.append(res)
            sample_recs[m.name] = {u: m.recommend(u, k=self.k) for u in sample_users}

        return PipelineOutput(results=results, sample_recs=sample_recs, models={m.name: m for m in models})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")
    out = Pipeline().run()
    print()
    print(out.table().to_string(index=False))
    print()
    for model_name, recs in out.sample_recs.items():
        print(f"\n=== {model_name} — sample recommendations ===")
        for uid, items in recs.items():
            print(f"  user {uid}: {items}")


if __name__ == "__main__":
    main()
