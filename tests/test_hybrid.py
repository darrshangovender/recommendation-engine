from __future__ import annotations

import numpy as np

from recommender.models import (
    CollabALSRecommender,
    ContentRecommender,
    HybridRecommender,
)


def test_alpha_one_matches_content_ranking(synthetic_data):
    """alpha=1 should rank candidates by content score alone."""
    ratings, items = synthetic_data["ratings"], synthetic_data["items"]
    content = ContentRecommender().fit(ratings, items=items)
    collab = CollabALSRecommender(factors=8, iterations=5).fit(ratings, items=items)
    hybrid = HybridRecommender(content=content, collab=collab).fit(ratings, items=items).with_fixed_alpha(1.0)

    candidates = [1, 2, 3, 4, 5, 6, 7, 8]
    blended = hybrid.score(user_id=1, item_ids=candidates)
    content_scores = content.score(user_id=1, item_ids=candidates)
    # alpha=1 -> blended is the min-max of content scores. Ranking must match.
    assert list(np.argsort(-blended)) == list(np.argsort(-content_scores))


def test_alpha_zero_matches_collab_ranking(synthetic_data):
    ratings, items = synthetic_data["ratings"], synthetic_data["items"]
    content = ContentRecommender().fit(ratings, items=items)
    collab = CollabALSRecommender(factors=8, iterations=5).fit(ratings, items=items)
    hybrid = HybridRecommender(content=content, collab=collab).fit(ratings, items=items).with_fixed_alpha(0.0)

    candidates = [1, 2, 3, 4, 5, 6, 7, 8]
    blended = hybrid.score(user_id=1, item_ids=candidates)
    collab_scores = collab.score(user_id=1, item_ids=candidates)
    assert list(np.argsort(-blended)) == list(np.argsort(-collab_scores))


def test_alpha_ramp_uses_history_depth(synthetic_data):
    """Cold users → alpha=1 (content). Warm users → alpha=warm_alpha."""
    ratings, items = synthetic_data["ratings"], synthetic_data["items"]
    hybrid = HybridRecommender(cold_threshold=2, warm_threshold=4, warm_alpha=0.2).fit(ratings, items=items)
    # Manually rewrite the event count map to control the test.
    hybrid._user_event_count = {1: 1, 2: 3, 3: 10}
    assert hybrid._alpha_for(1) == 1.0       # cold
    assert hybrid._alpha_for(3) == 0.2       # warm
    mid = hybrid._alpha_for(2)
    assert 0.2 < mid < 1.0                   # ramping
