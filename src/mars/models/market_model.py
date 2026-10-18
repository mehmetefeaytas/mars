# SPDX-License-Identifier: AGPL-3.0-only
"""Event Study / Market Model — endeks ayrismasi (abnormal return).

Akademik dayanak:
  * MacKinlay, A. C. (1997). Event Studies in Economics and Finance.
    Journal of Economic Literature 35(1), 13-39.

Model:
  R_i,t = alpha_i + beta_i * R_m,t + e_i,t
  AR_i,t = R_i,t - (alpha_i + beta_i * R_m,t)
  CAR_i(t1,t2) = sum_{t=t1}^{t2} AR_i,t

Endeksin kendi oynakligini ayristirmak icin ad-hoc "endeksten kopuş" yerine
market model artigi kullanilir. Artiklar robust z ile normalize edilir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..anomaly.robust_stats import robust_zscore, squash_to_unit


@dataclass
class MarketModelFit:
    alpha: float
    beta: float
    sigma: float
    n_obs: int


def fit_market_model(
    stock_ret: pd.Series, market_ret: pd.Series, estimation_window: int | None = None
) -> MarketModelFit:
    """OLS ile market model katsayilarini tahmin eder.

    estimation_window verilmezse tum orneklem kullanilir. Verilirse SON pencere
    (nedensellik icin cagiran taraf kaydirir).
    """
    df = pd.concat([stock_ret.rename("r"), market_ret.rename("m")], axis=1).dropna()
    if estimation_window is not None and len(df) > estimation_window:
        df = df.iloc[-estimation_window:]
    if len(df) < 10:
        return MarketModelFit(alpha=0.0, beta=1.0, sigma=float(df["r"].std(ddof=0) or 1e-4), n_obs=len(df))

    m = df["m"].to_numpy()
    r = df["r"].to_numpy()
    beta, alpha = np.polyfit(m, r, 1)
    resid = r - (alpha + beta * m)
    sigma = float(np.std(resid, ddof=0))
    return MarketModelFit(alpha=float(alpha), beta=float(beta), sigma=sigma, n_obs=len(df))


def abnormal_returns(
    stock_ret: pd.Series, market_ret: pd.Series, fit: MarketModelFit
) -> pd.Series:
    """AR_i,t = R_i,t - (alpha + beta R_m,t)."""
    aligned = pd.concat([stock_ret.rename("r"), market_ret.rename("m")], axis=1).dropna()
    expected = fit.alpha + fit.beta * aligned["m"]
    ar = aligned["r"] - expected
    ar.name = "abnormal_ret"
    return ar


def rolling_abnormal_returns(
    stock_ret: pd.Series, market_ret: pd.Series, window: int = 120, min_periods: int = 30
) -> pd.Series:
    """Rolling market model ile gunluk anormal getiri (AR).

    Her t gunu icin [t-window+1, t] penceresinde alpha/beta yeniden tahmin edilir
    (degisken beta / rejim dayanikliligi). Tahmin penceresi t'yi icerdigi icin
    sinyal t'nin kendi artigidir; gelecege bakmaz (nedensel).
    """
    df = pd.concat([stock_ret.rename("r"), market_ret.rename("m")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float, name="abnormal_ret")
    m = df["m"].to_numpy()
    r = df["r"].to_numpy()
    n = len(df)
    ar = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window + 1)
        mm, rr = m[lo : i + 1], r[lo : i + 1]
        if len(mm) < min_periods or np.std(mm) == 0:
            continue
        beta, alpha = np.polyfit(mm, rr, 1)
        ar[i] = r[i] - (alpha + beta * m[i])
    return pd.Series(ar, index=df.index, name="abnormal_ret")


def signal_from_ar(ar: pd.Series) -> pd.Series:
    """Anormal getiri serisinden [0,1] normalize ayrisim sinyali (robust z + sigmoid)."""
    values = ar.to_numpy(dtype=float)
    mask = np.isfinite(values)
    sig = np.full(len(values), np.nan)
    if mask.any():
        z = robust_zscore(values[mask])
        sig[mask] = squash_to_unit(z, k=1.0)
    return pd.Series(sig, index=ar.index, name="sig_index_divergence")


def index_divergence_signal(
    stock_ret: pd.Series,
    market_ret: pd.Series,
    *,
    rolling: int = 120,
    fit: MarketModelFit | None = None,
) -> pd.Series:
    """[0,1] normalize endeks ayrisim sinyali (yuksek = endeksten anormal kopuş).

    fit verilirse sabit market model artigi kullanilir; verilmezse KAYAN (rolling)
    market model ile her gunun artigi hesaplanir (bkz. rolling_abnormal_returns).
    """
    df = pd.concat([stock_ret.rename("r"), market_ret.rename("m")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float, name="sig_index_divergence")

    if fit is not None:
        ar = df["r"] - (fit.alpha + fit.beta * df["m"])
        ar = pd.Series(ar.to_numpy(), index=df.index)
    else:
        ar = rolling_abnormal_returns(df["r"], df["m"], window=rolling)

    return signal_from_ar(ar)


def cumulative_abnormal_return(ar: pd.Series, window: int = 5) -> pd.Series:
    """Son `window` gunun kumulatif anormal getirisi (CAR)."""
    return ar.rolling(window, min_periods=1).sum().rename("car")
