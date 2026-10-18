# SPDX-License-Identifier: AGPL-3.0-only
"""Ozellik muhendisligi — robust istatistik tabanli.

Akademik dayanaklar:
  * Robust olcekleme: median + MAD [Rousseeuw & Hubert 2011].
    Z_robust = (x - median) / (1.4826 * MAD),  MAD = median(|x - median|)
  * Likidite vekili: Amihud illiquidity [Amihud 2002].
    ILLIQ_t = |R_t| / (DollarVolume_t)
  * Log getiri ve normal aralik: standart finansal oznitelikler.

Tum ozellikler nedenseldir (yalnizca gecmis pencereyi kullanir) — ileriye bakmaz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings, load_settings

FEATURE_COLUMNS = [
    "log_ret",
    "vol_z",
    "norm_range",
    "volatility_z",
    "illiquidity_z",
    "clv",
]


def robust_z(series: pd.Series, scale: float = 1.4826) -> pd.Series:
    """Medyan/MAD tabanli robust z-skoru (tum seri uzerinde)."""
    x = pd.to_numeric(series, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    denom = scale * mad
    if not np.isfinite(denom) or denom == 0:
        # MAD sifir ise standart sapmaya dus; o da sifirsa sifir dondur.
        sd = x.std(ddof=0)
        denom = sd if np.isfinite(sd) and sd > 0 else 1.0
    return (x - med) / denom


def rolling_robust_z(
    series: pd.Series, window: int, scale: float = 1.4826, min_periods: int | None = None
) -> pd.Series:
    """Rolling (nedensel) robust z-skoru."""
    x = pd.to_numeric(series, errors="coerce")
    mp = min_periods if min_periods is not None else max(5, window // 3)
    med = x.rolling(window, min_periods=mp).median()
    mad = (x - med).abs().rolling(window, min_periods=mp).median()
    denom = scale * mad
    sd = x.rolling(window, min_periods=mp).std(ddof=0)
    denom = denom.where(denom > 0, sd)
    denom = denom.replace(0.0, np.nan)
    return (x - med) / denom


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def norm_range(df: pd.DataFrame) -> pd.Series:
    """Gun ici normal aralik: (high - low) / close."""
    return (df["high"] - df["low"]) / df["close"]


def amihud_illiquidity(df: pd.DataFrame) -> pd.Series:
    """Amihud (2002) gunluk likidite vekili: |log_ret| / dolar hacmi (olceklendi)."""
    ret = log_returns(df["close"]).abs()
    dollar_volume = df["close"] * df["volume"]
    dv = dollar_volume.replace(0.0, np.nan)
    return (ret / dv) * 1e9


def close_location_value(df: pd.DataFrame) -> pd.Series:
    """CLV: kapanisin gun ici araliktaki konumu, [-1, +1].

    +1 = high'da kapanis (kusefe basma / marking the close isareti),
    -1 = low'da kapanis. Manipulasyon literaturunde kapanis fiyati uzerinde
    oynama (closing price manipulation) klasik bir kalıntidir.
    """
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    return (2.0 * df["close"] - df["high"] - df["low"]) / rng


def dollar_volume(df: pd.DataFrame) -> pd.Series:
    return df["close"] * df["volume"]


def up_streak(ret: pd.Series) -> pd.Series:
    """Ardisik pozitif gun sayisi (pump scheme'lerin birikim evresi)."""
    pos = (ret > 0).astype(int)
    streak = pos.groupby((pos != pos.shift()).cumsum()).cumsum()
    return streak.where(pos == 1, 0)


def add_features(df: pd.DataFrame, settings: Settings | None = None) -> pd.DataFrame:
    """Ham OHLCV cercevesine robust ozellik sutunlarini ekler."""
    settings = settings or load_settings()
    w = settings.rolling_window
    out = df.copy()

    out["log_ret"] = log_returns(out["close"])
    out["vol_z"] = rolling_robust_z(out["volume"], w, settings.mad_scale)
    out["norm_range"] = norm_range(out)
    out["norm_range_z"] = rolling_robust_z(out["norm_range"], w, settings.mad_scale)
    out["volatility"] = out["log_ret"].rolling(w, min_periods=max(5, w // 3)).std(ddof=0)
    out["volatility_z"] = rolling_robust_z(out["volatility"], w, settings.mad_scale)
    out["illiquidity"] = amihud_illiquidity(out)
    out["illiquidity_z"] = rolling_robust_z(out["illiquidity"], w, settings.mad_scale)
    # Manipulasyon baglam etiketleri
    out["dollar_volume"] = dollar_volume(out)
    out["adv_dollar"] = out["dollar_volume"].rolling(w, min_periods=max(5, w // 3)).median()
    out["clv"] = close_location_value(out)
    out["clv_z"] = rolling_robust_z(out["clv"], w, settings.mad_scale)
    out["up_streak"] = up_streak(out["log_ret"])
    return out


def build_feature_matrix(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Ozellik matrisi X in R^{n x m} dondurur (NaN satirlari dusurulur)."""
    cols = columns or FEATURE_COLUMNS
    present = [c for c in cols if c in df.columns]
    X = df[present].dropna().to_numpy(dtype=float)
    return X, present
