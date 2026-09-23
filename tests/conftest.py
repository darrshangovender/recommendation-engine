"""Shared test fixtures.

Every fixture here is hermetic: no network, no cached downloads, no compiled
optional dependencies. Tests that need MovieLens get a *synthetic* archive
written to ``tmp_path`` in the real ml-100k on-disk format and parsed by the
real loader, so the parsing logic stays covered without a 5MB download inside
CI. The genuine download is exercised only by the opt-in ``network``-marked
test in ``test_data.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recommender.data.loader import GENRES


@pytest.fixture
def synthetic_data() -> dict[str, pd.DataFrame]:
    """A tiny dataset: 6 users, 8 items, 3 genres, ~30 events."""
    rng = np.random.default_rng(0)
    items = pd.DataFrame({
        "item_id": list(range(1, 9)),
        "title": [f"Movie {i}" for i in range(1, 9)],
        "genres": [
            ["Action"], ["Action", "Sci-Fi"], ["Drama"], ["Drama", "Romance"],
            ["Comedy"], ["Comedy", "Romance"], ["Action", "Drama"], ["Sci-Fi"],
        ],
    })
    users = list(range(1, 7))
    rows = []
    base_ts = 1_000_000
    for u in users:
        # Each user rates 4-6 items.
        n = rng.integers(4, 7)
        chosen = rng.choice(items["item_id"], size=n, replace=False)
        for i, iid in enumerate(chosen):
            rows.append({
                "user_id": int(u),
                "item_id": int(iid),
                "rating": float(rng.integers(1, 6)),
                "timestamp": base_ts + u * 1000 + i,
            })
    ratings = pd.DataFrame(rows).astype({"user_id": np.int32, "item_id": np.int32, "rating": np.float32, "timestamp": np.int64})
    return {"ratings": ratings, "items": items}


# --- Synthetic MovieLens ---------------------------------------------------
#
# Shape of the fake dataset, asserted on by tests/test_data.py.
FAKE_N_USERS = 20
FAKE_N_ITEMS = 15
FAKE_RATINGS_PER_USER = 5
FAKE_N_RATINGS = FAKE_N_USERS * FAKE_RATINGS_PER_USER


def _write_fake_ml100k(root: Path) -> Path:
    """Write a miniature ml-100k directory in the exact upstream file format.

    Reproduces the three quirks the loader actually has to cope with: the
    tab-separated ``u.data``, the pipe-separated ``u.item`` with 19 trailing
    one-hot genre flags, and latin-1 encoded titles.
    """
    data_dir = root / "ml-100k"
    data_dir.mkdir(parents=True, exist_ok=True)

    # u.data — user_id \t item_id \t rating \t timestamp
    lines = []
    for user in range(1, FAKE_N_USERS + 1):
        for j in range(FAKE_RATINGS_PER_USER):
            item = ((user + j) % FAKE_N_ITEMS) + 1
            rating = ((user + j) % 5) + 1
            timestamp = 874_000_000 + user * 1000 + j
            lines.append(f"{user}\t{item}\t{rating}\t{timestamp}")
    (data_dir / "u.data").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # u.item — id|title|release|video_release|imdb_url| + 19 genre flags.
    # Item 1 carries no genre flags at all, mirroring upstream rows that are
    # tagged only "unknown"; the loader must still produce a list for it.
    item_lines = []
    for item in range(1, FAKE_N_ITEMS + 1):
        flags = [0] * len(GENRES)
        if item > 1:
            flags[1 + (item % (len(GENRES) - 1))] = 1
            if item % 3 == 0:
                flags[1 + ((item + 4) % (len(GENRES) - 1))] = 1
        # A non-ASCII title proves the latin-1 decode path.
        title = f"Fake Movie {item} (Café)" if item == 2 else f"Fake Movie {item} (1995)"
        fields = [
            str(item), title, "01-Jan-1995", "",
            f"http://example.invalid/movie{item}",
            *[str(f) for f in flags],
        ]
        item_lines.append("|".join(fields))
    (data_dir / "u.item").write_text("\n".join(item_lines) + "\n", encoding="latin-1")

    # u.user — user_id|age|gender|occupation|zip
    user_lines = [
        f"{u}|{20 + (u % 40)}|{'M' if u % 2 else 'F'}|engineer|{7000 + u}"
        for u in range(1, FAKE_N_USERS + 1)
    ]
    (data_dir / "u.user").write_text("\n".join(user_lines) + "\n", encoding="utf-8")
    return data_dir


@pytest.fixture
def fake_movielens_cache(tmp_path: Path) -> Path:
    """A cache dir already populated with the synthetic archive.

    Because ``ml-100k/u.data`` exists, the loader's download step short-circuits
    — so this never touches the network even though it runs the real code path.
    """
    _write_fake_ml100k(tmp_path)
    return tmp_path


@pytest.fixture
def movielens(fake_movielens_cache: Path):
    """Synthetic MovieLens parsed by the real loader. Offline and deterministic."""
    from recommender.data.loader import load_movielens_100k

    return load_movielens_100k(cache_dir=fake_movielens_cache)


@pytest.fixture(scope="session")
def movielens_live():
    """The genuine MovieLens-100K download. Opt-in only.

    Guarded twice over: the consuming test carries the ``network`` marker, and
    this still skips rather than fails if the download is unavailable.
    """
    if os.environ.get("RECOMMENDER_NETWORK_TESTS") != "1":
        pytest.skip("network tests disabled; set RECOMMENDER_NETWORK_TESTS=1 to enable")
    from recommender.data.loader import load_movielens_100k

    try:
        return load_movielens_100k()
    except Exception as exc:  # noqa: BLE001  network down, mirror moved, etc. -> skip, never fail
        pytest.skip(f"MovieLens not available: {exc}")
