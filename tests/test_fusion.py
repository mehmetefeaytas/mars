# SPDX-License-Identifier: AGPL-3.0-only
"""Füzyon, kanit (D-S) ve haber duzeltmesi testleri."""

import numpy as np
import pandas as pd

from mars.config import FusionWeights
from mars.fusion.weighted import fuse
from mars.fusion.evidence import combine_two, ds_belief_anomaly, signal_to_bba, fuse_evidence
from mars.news.sources import NewsItem
from mars.news.analyzer import analyze_news
from mars.signals import SIGNAL_COLUMNS


def _signal_frame(vals: dict[str, float], n: int = 5) -> pd.DataFrame:
    data = {c: np.full(n, np.nan) for c in SIGNAL_COLUMNS}
    for k, v in vals.items():
        data[k] = np.full(n, v)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(data, index=idx)


def test_weights_normalize():
    w = FusionWeights(volume_z=2, index_divergence=2, volatility=0, mahalanobis=0,
                      isolation_forest=0, liquidity=0)
    nw = w.normalized()
    assert abs(sum(nw.values()) - 1.0) < 1e-9
    assert nw["volume_z"] > 0


def test_fuse_zero_signals_zero_suspicion():
    sf = _signal_frame({c: 0.0 for c in SIGNAL_COLUMNS})
    f = fuse(sf)
    assert np.allclose(f.frame["suspicion"], 0.0)


def test_fuse_max_signals_high_suspicion():
    sf = _signal_frame({c: 1.0 for c in SIGNAL_COLUMNS})
    f = fuse(sf)
    assert f.frame["suspicion"].iloc[0] > 99.0


def test_contributions_sum_to_suspicion():
    sf = _signal_frame({c: v for c, v in zip(SIGNAL_COLUMNS, np.linspace(0.1, 0.9, len(SIGNAL_COLUMNS)))})
    f = fuse(sf)
    contrib_cols = [f"contrib_{c}" for c in SIGNAL_COLUMNS]
    assert np.isclose(f.frame[contrib_cols].sum(axis=1).iloc[0],
                      f.frame["suspicion"].iloc[0], atol=1e-6)


def test_ds_combination_conflict_neutral():
    m = combine_two({"A": 1.0, "N": 0.0, "U": 0.0}, {"A": 0.0, "N": 1.0, "U": 0.0})
    assert m["A"] == 0.5  # tam celiski -> notr


def test_ds_belief_agreement():
    m = signal_to_bba(1.0, 1.0)
    assert ds_belief_anomaly([m, m]) > 0.99


def test_ds_fuse_runs():
    sf = _signal_frame({c: 0.8 for c in SIGNAL_COLUMNS}, n=10)
    s = fuse_evidence(sf)
    assert (s > 90).all()


def test_news_reduces_suspicion_when_confirmed():
    sf = _signal_frame({c: 1.0 for c in SIGNAL_COLUMNS}, n=3)
    idx = sf.index[0]
    ar = pd.Series(0.08, index=sf.index)  # pozitif hareket
    news = {idx: [analyze_news(NewsItem(headline="Şirket güçlü kâr açıkladı",
                                        source="kap", official=True))]}
    f_no = fuse(sf, abnormal_returns=ar)
    f_yes = fuse(sf, abnormal_returns=ar, news_by_date=news)
    assert f_yes.frame["suspicion"].loc[idx] < f_no.frame["suspicion"].loc[idx]
    assert f_yes.frame["news_factor"].loc[idx] < 1.0
