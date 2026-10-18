# SPDX-License-Identifier: AGPL-3.0-only
"""Uctan uca pipeline testleri (aga cikmadan, sentetik veri ile)."""

import numpy as np
import pandas as pd

from mars.config import load_settings
from mars.fusion.weighted import fuse
from mars.ingestion import align_stock_market
from mars.models.market_model import rolling_abnormal_returns
from mars.signals import SIGNAL_COLUMNS, compute_signals
from mars.synthetic import SyntheticConfig, generate_market, generate_stock


def _analysis(seed=11, kind="pump"):
    settings = load_settings()
    cfg = SyntheticConfig(n_days=400, seed=seed)
    stock = generate_stock(cfg, n_injections=6, injection_kind=kind)
    market = generate_market(cfg)
    sf = compute_signals(align_stock_market(stock, market), settings)
    ar = rolling_abnormal_returns(sf.frame["log_ret"], sf.frame["market_log_ret"],
                                  window=settings.estimation_window)
    f = fuse(sf.frame, abnormal_returns=ar, settings=settings)
    return sf, f, stock


def test_signals_bounded():
    sf, _, _ = _analysis()
    vals = sf.frame[SIGNAL_COLUMNS].to_numpy()
    finite = vals[np.isfinite(vals)]
    assert finite.min() >= -1e-9
    assert finite.max() <= 1.0 + 1e-9


def test_injections_have_higher_suspicion():
    sf, f, stock = _analysis()
    inj = [i for i in stock.attrs["injection_idx"] if i < len(f.frame)]
    assert f.frame["suspicion"].iloc[inj].mean() > f.frame["suspicion"].mean() + 20


def test_suspicion_range():
    _, f, _ = _analysis()
    s = f.frame["suspicion"]
    assert s.min() >= 0.0
    assert s.max() <= 100.0


def test_data_confidence_range():
    _, f, _ = _analysis()
    dc = f.frame["data_confidence"]
    assert dc.min() >= 0
    assert dc.max() <= 100


def test_subtle_scenario_multidimensional_beats_volume():
    # Ince senaryoda cok boyutlu skor, tekil hacim sinyalinden daha ayirt edici olmali.
    from sklearn.metrics import roc_auc_score
    sf, f, stock = _analysis(seed=5, kind="subtle")
    inj = [i for i in stock.attrs["injection_idx"] if i < len(f.frame)]
    labels = np.zeros(len(f.frame), dtype=int)
    labels[inj] = 1
    auc_volume = roc_auc_score(labels, sf.frame["sig_volume"].fillna(0.0))
    auc_fusion = roc_auc_score(labels, f.frame["suspicion"])
    assert auc_fusion >= auc_volume
