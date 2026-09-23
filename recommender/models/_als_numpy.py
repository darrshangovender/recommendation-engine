"""Pure-NumPy implicit-feedback ALS — the fallback when ``implicit`` is absent.

Why this exists
---------------
``implicit`` ships compiled extensions and is a frequent source of install
failures on CI images and on Python versions newer than its latest wheel. Making
it a hard requirement meant the *entire* collaborative-filtering test suite went
uncollectable whenever the wheel was unavailable. A ~60-line NumPy solver keeps
that behaviour under test everywhere, and ``implicit`` stays the default when it
is installed because it is dramatically faster on real-sized matrices.

Algorithm
---------
Alternating Least Squares for implicit feedback, Hu/Koren/Volinsky (2008). For
each user ``u`` with observed items ``I_u``::

    c_u = 1 + alpha * r_u                    (confidence)
    p_u = 1                                  (preference, binary)
    x_u = (YᵀY + Yᵤᵀ(C_u - I)Yᵤ + λI)⁻¹ Yᵤᵀ C_u p_u

and symmetrically for items. Precomputing ``YᵀY`` once per half-iteration is
what keeps this O(nnz · f²) rather than O(users · items · f²).

This intentionally implements only the slice of the ``implicit`` API that
:class:`~recommender.models.collab_als.CollabALSRecommender` consumes, so the
two are drop-in interchangeable.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


class NumpyALS:
    """Drop-in stand-in for ``implicit.als.AlternatingLeastSquares``.

    Only the attributes and methods used by the recommender are provided:
    ``fit``, ``recommend``, ``user_factors``, ``item_factors``.
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        iterations: int = 15,
        alpha: float = 20.0,
        random_state: int = 42,
        **_ignored: object,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state
        self.user_factors: np.ndarray = np.empty((0, factors), dtype=np.float32)
        self.item_factors: np.ndarray = np.empty((0, factors), dtype=np.float32)

    def fit(self, user_items: csr_matrix, show_progress: bool = False) -> NumpyALS:
        """Factorise ``user_items`` (users x items) into latent factors."""
        n_users, n_items = user_items.shape
        rng = np.random.default_rng(self.random_state)
        # Small random init: exact zeros would leave the first solve singular
        # in the regularisation-free directions.
        self.user_factors = (rng.normal(0, 0.01, (n_users, self.factors))).astype(np.float32)
        self.item_factors = (rng.normal(0, 0.01, (n_items, self.factors))).astype(np.float32)

        user_rows = user_items.tocsr()
        item_rows = user_items.T.tocsr()

        for _ in range(self.iterations):
            self.user_factors = self._solve_side(user_rows, self.item_factors)
            self.item_factors = self._solve_side(item_rows, self.user_factors)
        return self

    def _solve_side(self, rows: csr_matrix, other: np.ndarray) -> np.ndarray:
        """Solve one half-iteration: refit every row of ``rows`` against ``other``."""
        n_rows = rows.shape[0]
        f = self.factors
        # Shared across all rows — the expensive part, computed once.
        gram = other.T @ other
        reg = self.regularization * np.eye(f, dtype=np.float64)
        out = np.zeros((n_rows, f), dtype=np.float32)

        indptr, indices, data = rows.indptr, rows.indices, rows.data
        for i in range(n_rows):
            start, end = indptr[i], indptr[i + 1]
            if start == end:
                continue  # no observations — leave the row at zero
            cols = indices[start:end]
            confidence = 1.0 + self.alpha * data[start:end].astype(np.float64)
            sub = other[cols].astype(np.float64)  # (n_obs, f)
            # A = gram + subᵀ(C-I)sub + λI ; b = subᵀ C p, with p == 1.
            a = gram + sub.T @ ((confidence - 1.0)[:, None] * sub) + reg
            b = sub.T @ confidence
            out[i] = np.linalg.solve(a, b).astype(np.float32)
        return out

    def recommend(
        self,
        userid: int,
        user_items: csr_matrix,
        N: int = 10,
        filter_already_liked_items: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-``N`` item indices and scores for one user.

        ``user_items`` is that user's single row, matching ``implicit``'s
        signature.
        """
        scores = (self.item_factors @ self.user_factors[userid]).astype(np.float64)
        if filter_already_liked_items:
            seen = np.asarray(user_items.indices if hasattr(user_items, "indices") else [])
            if seen.size:
                scores[seen] = -np.inf

        n = int(min(N, np.sum(np.isfinite(scores))))
        if n <= 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        # argpartition for the top-n, then sort just those descending.
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        return top.astype(np.int64), scores[top]
