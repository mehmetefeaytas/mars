# SPDX-License-Identifier: AGPL-3.0-only
"""Sentetik OHLCV uretici.

Amac: (i) birim testleri aga cikmadan kosturmak, (ii) etiketli zemin gercegi
olmadigindan denetimsiz motoru dogrulamak icin bilinen manipulasyon/olay
senaryolari uretmek. Uretilen veri gercek piyasa verisi DEGILDIR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Settings


@dataclass
class SyntheticConfig:
    n_days: int = 300
    start: str = "2023-01-02"
    seed: int = 7
    base_price: float = 100.0
    base_volume: float = 1_000_000.0
    daily_vol: float = 0.015
    market_vol: float = 0.010


def _business_days(n: int, start: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def generate_market(cfg: SyntheticConfig) -> pd.DataFrame:
    """Piyasa (endeks) getirileri — ortak faktör."""
    rng = np.random.default_rng(cfg.seed)
    dates = _business_days(cfg.n_days, cfg.start)
    ret = rng.normal(0.0, cfg.market_vol, len(dates))
    close = 1000.0 * np.exp(np.cumsum(ret))
    return pd.DataFrame({"date": dates, "close": close, "log_ret": ret}).set_index("date")


def generate_stock(
    cfg: SyntheticConfig,
    beta: float = 1.0,
    *,
    n_injections: int = 0,
    injection_kind: str = "pump",
) -> pd.DataFrame:
    """Tek hisse OHLCV uretir; opsiyonel olarak bilinen anomali gunleri enjekte eder.

    injection_kind:
      - "pump": haber/hacim destegi olmadan hacim+getiri sicramasi (supheli)
      - "news_legit": buyuk ama hacimle ve oynaklikla tutarli mesru sicrama
    """
    rng = np.random.default_rng(cfg.seed + int(beta * 10))
    dates = _business_days(cfg.n_days, cfg.start)
    market = generate_market(cfg)["log_ret"].to_numpy()
    idio = rng.normal(0.0, cfg.daily_vol, len(dates))
    ret = beta * market + idio

    volume = np.abs(rng.normal(cfg.base_volume, cfg.base_volume * 0.2, len(dates)))

    injection_idx = np.array([], dtype=int)
    if n_injections > 0:
        lo, hi = int(0.3 * len(dates)), int(0.9 * len(dates))
        injection_idx = rng.choice(np.arange(lo, hi), size=n_injections, replace=False)
        for i in injection_idx:
            if injection_kind == "pump":
                ret[i] += rng.uniform(0.06, 0.12)
                volume[i] *= rng.uniform(4.0, 8.0)
            elif injection_kind == "news_legit":
                ret[i] += rng.uniform(0.05, 0.10)
                volume[i] *= rng.uniform(1.8, 2.6)

    close = cfg.base_price * np.exp(np.cumsum(ret))
    prev_close = np.concatenate([[cfg.base_price], close[:-1]])
    high = np.maximum(close, prev_close) * (1.0 + np.abs(rng.normal(0, 0.004, len(dates))))
    low = np.minimum(close, prev_close) * (1.0 - np.abs(rng.normal(0, 0.004, len(dates))))
    open_ = prev_close * (1.0 + rng.normal(0, 0.003, len(dates)))

    df = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    ).set_index("date")
    df.attrs["injection_idx"] = injection_idx
    df.attrs["injection_kind"] = injection_kind
    return df


def default_settings() -> Settings:
    return Settings()
