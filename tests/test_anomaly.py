# SPDX-License-Identifier: AGPL-3.0-only
"""Anomali cekirdegi testleri (MCD-Mahalanobis, iForest, market model, EVT, BOCPD)."""

import numpy as np
import pandas as pd

from mars.anomaly.iforest import fit_isolation_forest
from mars.anomaly.robust_stats import robust_mahalanobis, squash_to_unit
from mars.anomaly.evt import evt_signal, fit_pot, tail_exceedance_probability
from mars.anomaly.changepoint import bocpd_changepoint_prob
from mars.models.market_model import fit_market_model, abnormal_returns, rolling_abnormal_returns


def test_mahalanobis_flags_outlier():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(200, 3))
    X = np.vstack([X, [12.0, 12.0, 12.0]])
    res = robust_mahalanobis(X)
    assert res.anomaly[-1] > 0.99
    assert res.anomaly[:-1].mean() < 0.6


def test_iforest_flags_outlier():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(300, 2))
    X = np.vstack([X, [15.0, 15.0]])
    res = fit_isolation_forest(X)
    assert res.score[-1] > 0.9


def test_squash_bounds():
    z = np.array([-100, 0, 3, 100])
    s = squash_to_unit(z)
    assert np.all((s >= 0) & (s <= 1))
    assert s[0] < s[2] < s[3]


def test_market_model_beta_recovery():
    rng = np.random.default_rng(2)
    m = pd.Series(rng.normal(0, 0.01, 300))
    r = 2.0 * m + pd.Series(rng.normal(0, 0.001, 300))
    fit = fit_market_model(r, m)
    assert abs(fit.beta - 2.0) < 0.1
    ar = abnormal_returns(r, m, fit)
    assert ar.abs().mean() < 0.005


def test_rolling_ar_is_causal_lookahead_free():
    rng = np.random.default_rng(3)
    m = pd.Series(rng.normal(0, 0.01, 200))
    r = 1.2 * m + rng.normal(0, 0.005, 200)
    ar = rolling_abnormal_returns(r, m, window=60, min_periods=30)
    assert ar.iloc[:29].isna().all()
    assert np.isfinite(ar.iloc[-1])


def test_evt_tail_probability_decreases():
    rng = np.random.default_rng(4)
    losses = -rng.standard_t(df=3, size=500) * 0.02
    fit = fit_pot(losses, 0.9)
    assert fit.valid
    p_small = tail_exceedance_probability(fit.threshold * 1.1, fit)
    p_large = tail_exceedance_probability(fit.threshold * 3.0, fit)
    assert p_large < p_small


def test_evt_signal_extreme_spike():
    rng = np.random.default_rng(5)
    ret = pd.Series(rng.standard_t(df=4, size=400) * 0.02)
    ret.iloc[100] = -0.15
    sig = evt_signal(ret)
    assert sig.iloc[100] > 0.5


def test_bocpd_detects_mean_shift():
    rng = np.random.default_rng(6)
    x = np.concatenate([rng.normal(0, 0.01, 200), rng.normal(0.15, 0.01, 200)])
    cp = bocpd_changepoint_prob(x, hazard=1 / 100)
    assert cp[200] > 0.5
    assert cp[:150].mean() < 0.1
