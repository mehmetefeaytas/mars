# SPDX-License-Identifier: AGPL-3.0-only
"""Ozellik muhendisligi testleri (robust istatistik, nedensellik)."""

import numpy as np
import pandas as pd

from mars.features import (
    add_features,
    build_feature_matrix,
    log_returns,
    robust_z,
    rolling_robust_z,
)
from mars.synthetic import SyntheticConfig, generate_market, generate_stock
from mars.ingestion import align_stock_market


def _aligned():
    cfg = SyntheticConfig(n_days=200, seed=3)
    stock = generate_stock(cfg)
    market = generate_market(cfg)
    return align_stock_market(stock, market)


def test_robust_z_is_resistant_to_outliers():
    rng = np.random.default_rng(0)
    x = pd.Series(np.r_[rng.normal(0, 1, 200), 1e6])
    z = robust_z(x)
    # Tek dev aykiri deger medyan/MAD'i bozmamali; normal noktalar kucuk, aykiri dev.
    assert abs(z.iloc[50]) < 10
    assert z.iloc[-1] > 1e5


def test_log_returns_identity():
    close = pd.Series([100.0, 110.0])
    r = log_returns(close)
    assert np.isclose(r.iloc[1], np.log(1.1))


def test_rolling_robust_z_is_causal():
    s = pd.Series(np.arange(100.0))
    z = rolling_robust_z(s, window=20, min_periods=5)
    # Pencerenin basinda NaN, sonra art arda deger.
    assert z.iloc[:4].isna().all()
    assert z.iloc[-1] == z.iloc[-1]  # NaN degil


def test_add_features_columns():
    df = add_features(_aligned())
    for col in ["log_ret", "vol_z", "norm_range", "volatility_z", "illiquidity_z"]:
        assert col in df.columns


def test_feature_matrix_drops_nan():
    df = add_features(_aligned())
    X, cols = build_feature_matrix(df)
    assert X.ndim == 2
    assert not np.isnan(X).any()
    assert len(cols) == X.shape[1]
