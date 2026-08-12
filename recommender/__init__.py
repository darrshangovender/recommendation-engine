"""Hybrid recommendation engine reference implementation.

Public API:
    Recommender — abstract interface implemented by every model.
    Pipeline    — train → evaluate → emit recommendations.
"""

from recommender.models.base import Recommender
from recommender.pipeline import Pipeline

__all__ = ["Recommender", "Pipeline"]
__version__ = "0.1.0"
