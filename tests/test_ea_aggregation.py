"""unit tests for the min-fdr aggregator (F1 dedup fix)."""

import pytest

from gpath2vec.ea import aggregate_min_fdr


def test_min_fdr_picked_long_form():
    # niche A has fdr=0.5 for pathway X, niche B has fdr=0.001 for X
    records = [
        {"stId": "R-HSA-X", "fdr_bh": 0.5, "cluster": "A"},
        {"stId": "R-HSA-X", "fdr_bh": 0.001, "cluster": "B"},
    ]
    out = aggregate_min_fdr(records)
    assert len(out) == 1
    assert out[0]["stId"] == "R-HSA-X"
    assert out[0]["entities"]["fdr"] == pytest.approx(0.001)


def test_min_fdr_picked_long_form_reverse_order():
    # iteration order should not matter
    records = [
        {"stId": "R-HSA-X", "fdr_bh": 0.001, "cluster": "B"},
        {"stId": "R-HSA-X", "fdr_bh": 0.5, "cluster": "A"},
    ]
    out = aggregate_min_fdr(records)
    assert out[0]["entities"]["fdr"] == pytest.approx(0.001)


def test_min_fdr_nested_form():
    # Net.enrichment shape with nested entities.fdr
    records = [
        {"stId": "R-HSA-Y", "entities": {"fdr": 0.3}},
        {"stId": "R-HSA-Y", "entities": {"fdr": 0.01}},
    ]
    out = aggregate_min_fdr(records)
    assert len(out) == 1
    assert out[0]["entities"]["fdr"] == pytest.approx(0.01)


def test_multiple_pathways_independent():
    records = [
        {"stId": "R-HSA-X", "fdr_bh": 0.5},
        {"stId": "R-HSA-Y", "fdr_bh": 0.001},
        {"stId": "R-HSA-X", "fdr_bh": 0.05},
        {"stId": "R-HSA-Y", "fdr_bh": 0.2},
    ]
    out = {r["stId"]: r["entities"]["fdr"] for r in aggregate_min_fdr(records)}
    assert out["R-HSA-X"] == pytest.approx(0.05)
    assert out["R-HSA-Y"] == pytest.approx(0.001)


def test_empty_input():
    assert aggregate_min_fdr([]) == []


def test_missing_fdr_defaults_to_one():
    # a record with neither fdr_bh nor entities.fdr should default to 1.0 (notsig)
    records = [{"stId": "R-HSA-Z"}]
    out = aggregate_min_fdr(records)
    assert out[0]["entities"]["fdr"] == 1.0


def test_pandas_series_compatible():
    # ea_df.iterrows() yields (index, Series); Series.get(key, default) works.
    import pandas as pd
    df = pd.DataFrame([
        {"stId": "R-HSA-X", "fdr_bh": 0.5},
        {"stId": "R-HSA-X", "fdr_bh": 0.01},
    ])
    out = aggregate_min_fdr([row for _, row in df.iterrows()])
    assert out[0]["entities"]["fdr"] == pytest.approx(0.01)