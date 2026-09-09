"""unit tests for aucell.topk_per_niche selection modes.

verifies that:
  - "none" selects by absolute score and is captured by a shared housekeeping
    pathway high in every niche,
  - "zscore" selects by cross-niche relative elevation and escapes that capture,
    routing each niche to its distinctive pathway,
  - edge weights are the RAW aucell score in both modes,
  - zero-score pathways are never connected,
  - a single-niche cohort falls back to absolute selection with a warning,
  - an unknown mode raises.

run: pytest tests/test_aucell_topk.py -v
"""

import pandas as pd
import pytest

from gpath2vec.aucell import topk_per_niche


def _cohort():
    # HK high in every niche (shared); A, B distinctive to n0, n1.
    return pd.DataFrame(
        {"HK": [0.95, 0.96, 0.94],
         "A":  [0.80, 0.40, 0.42],
         "B":  [0.41, 0.82, 0.40]},
        index=["n0", "n1", "n2"])


def test_none_is_captured_by_housekeeping():
    clusters = topk_per_niche(_cohort(), k=1, standardize="none")
    assert all("HK" in d for d in clusters.values())


def test_zscore_escapes_housekeeping():
    clusters = topk_per_niche(_cohort(), k=1, standardize="zscore")
    assert all("HK" not in d for d in clusters.values())
    assert "A" in clusters["n0"]
    assert "B" in clusters["n1"]


def test_edge_weight_is_raw_score():
    df = _cohort()
    clusters = topk_per_niche(df, k=1, standardize="zscore")
    # weight is the raw aucell score, not the z-score
    assert clusters["n0"]["A"] == pytest.approx(df.loc["n0", "A"])


def test_default_is_none():
    df = _cohort()
    assert topk_per_niche(df, k=1) == topk_per_niche(df, k=1, standardize="none")


def test_drops_zero_score_pathways():
    df = pd.DataFrame({"A": [0.0, 0.5], "B": [0.7, 0.0]}, index=["n0", "n1"])
    clusters = topk_per_niche(df, k=5, standardize="none")
    assert "A" not in clusters["n0"]   # zero score never connected
    assert clusters["n0"] == {"B": 0.7}


def test_single_niche_falls_back_with_warning():
    df = pd.DataFrame({"A": [0.8], "B": [0.6]}, index=["n0"])
    with pytest.warns(RuntimeWarning):
        clusters = topk_per_niche(df, k=1, standardize="zscore")
    assert "A" in clusters["n0"]       # absolute fallback picks highest raw


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        topk_per_niche(_cohort(), k=1, standardize="bogus")