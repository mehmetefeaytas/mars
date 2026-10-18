# SPDX-License-Identifier: AGPL-3.0-only
"""Sinyal montaji — ham ozelliklerden normalize anomali sinyallerine.

Cikti: her gun icin [0,1] aralığında normalize edilmis sinyaller. Yuksek deger
= daha anormal. Bu sinyaller füzyon katmaninda agirlikli olarak birlestirilir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .anomaly.iforest import fit_isolation_forest
from .anomaly.robust_stats import robust_mahalanobis, robust_zscore, squash_to_unit
from .config import Settings, load_settings
from .features import FEATURE_COLUMNS, add_features
from .models.market_model import index_divergence_signal

SIGNAL_COLUMNS = [
    "sig_volume",
    "sig_volatility",
    "sig_liquidity",
    "sig_index_divergence",
    "sig_mahalanobis",
    "sig_isolation_forest",
]


@dataclass
class SignalFrame:
    frame: pd.DataFrame          # date index + sinyaller + destekleyici ham degerler
    feature_columns: list[str]

    def signal_matrix(self) -> np.ndarray:
        return self.frame[SIGNAL_COLUMNS].to_numpy(dtype=float)


def _squash_series(s: pd.Series, k: float = 1.0) -> pd.Series:
    vals = s.to_numpy(dtype=float)
    mask = np.isfinite(vals)
    out = np.full(len(vals), np.nan)
    if mask.any():
        out[mask] = squash_to_unit(robust_zscore(vals[mask]), k=k)
    return pd.Series(out, index=s.index)


def compute_signals(
    aligned: pd.DataFrame,
    settings: Settings | None = None,
) -> SignalFrame:
    """Hisse+endeks cercevesinden normalisan sinyal cercevesi uretir.

    `aligned`, ingestion.align_stock_market cikisi gibi `market_close` sutunu
    iceren ham OHLCV cercevesi olmalidir.
    """
    settings = settings or load_settings()
    df = add_features(aligned, settings)
    df["market_log_ret"] = np.log(df["market_close"] / df["market_close"].shift(1))

    # Tek degiskenli robust sinyaller
    df["sig_volume"] = _squash_series(df["vol_z"])
    df["sig_volatility"] = _squash_series(df["volatility_z"])
    df["sig_liquidity"] = _squash_series(df["illiquidity_z"])
    df["sig_index_divergence"] = index_divergence_signal(
        df["log_ret"], df["market_log_ret"], rolling=settings.estimation_window
    )

    # Cok degiskenli robust (MCD-Mahalanobis) — ozellik matrisi uzerinde
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    valid = df[feature_cols].dropna()
    maha = pd.Series(np.nan, index=df.index, name="sig_mahalanobis")
    if len(valid) >= len(feature_cols) + 2:
        res = robust_mahalanobis(valid.to_numpy())
        maha.loc[valid.index] = res.anomaly
    df["sig_mahalanobis"] = maha

    # Isolation Forest — ayni ozellik matrisi uzerinde
    iforest = pd.Series(np.nan, index=df.index, name="sig_isolation_forest")
    if len(valid) >= len(feature_cols) + 2:
        res_if = fit_isolation_forest(
            valid.to_numpy(),
            n_estimators=settings.iforest_trees,
            contamination=settings.iforest_contamination,
            random_state=settings.random_state,
        )
        iforest.loc[valid.index] = res_if.score
    df["sig_isolation_forest"] = iforest

    return SignalFrame(frame=df, feature_columns=feature_cols)
