# SPDX-License-Identifier: AGPL-3.0-only
"""Extreme Value Theory — Peaks Over Threshold (GPD).

Akademik dayanak:
  * McNeil, A. J., Frey, R. (2000). Estimation of tail-related risk measures for
    heteroscedastic financial time series. Journal of Empirical Finance 7(3).

Fikir: finansal getiriler kalin kuyrukludur (yuksek kurtosis). Gauss/MAD esikleri
kuyrugu hafife alir. Esigi u asan kayiplar (asiri degerler) Genellestirilmis Pareto
Dagilimi (GPD) ile modellenir:
    P(X - u > y | X > u) = (1 + xi y / sigma)^{-1/xi}
Bu, kuyruk olasiligi tabanli, kalibre edilmis bir anomali esigi saglar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class EVTFit:
    threshold: float
    xi: float          # sekil parametresi
    sigma: float       # olcek parametresi
    n_exceedances: int
    valid: bool


def fit_pot(losses: np.ndarray, threshold_q: float = 0.90) -> EVTFit:
    """Kayiplar uzerinde esik-asan GPD uydurur (floc=0)."""
    x = np.asarray(losses, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return EVTFit(threshold=np.inf, xi=0.0, sigma=1.0, n_exceedances=0, valid=False)
    u = float(np.quantile(x, threshold_q))
    excess = x[x > u] - u
    if len(excess) < 10:
        return EVTFit(threshold=u, xi=0.0, sigma=1.0, n_exceedances=len(excess), valid=False)
    try:
        xi, _, sigma = stats.genpareto.fit(excess, floc=0.0)
    except Exception:  # noqa: BLE001
        return EVTFit(threshold=u, xi=0.0, sigma=float(np.std(excess) or 1.0),
                      n_exceedances=len(excess), valid=False)
    return EVTFit(threshold=u, xi=float(xi), sigma=float(sigma), n_exceedances=len(excess), valid=True)


def tail_exceedance_probability(value: float, fit: EVTFit) -> float:
    """P(X > value | X > u) — ust kuyruk asma olasiligi (0..1)."""
    if not fit.valid or value <= fit.threshold:
        return 0.0
    y = value - fit.threshold
    xi, sigma = fit.xi, fit.sigma
    if abs(xi) < 1e-9:
        return float(np.exp(-y / sigma))
    base = 1.0 + xi * y / sigma
    if base <= 0:
        return 0.0
    return float(base ** (-1.0 / xi))


def evt_signal(log_ret: pd.Series, threshold_q: float = 0.90) -> pd.Series:
    """[0,1] EVT kuyruk anomali sinyali (yuksek = asiri kuyruk olayi).

    Tum orneklem uzerinde esik uydurulur; her gunun ust kuyruk olasiligi 1-p
    olarak olceklenir. Not: orneklem ici kalibrasyondur (ex-ante degil).
    """
    x = log_ret.to_numpy(dtype=float)
    losses = -x  # kayip yonu
    finite = losses[np.isfinite(losses)]
    if len(finite) < 30:
        return pd.Series(np.nan, index=log_ret.index, name="sig_evt")
    fit = fit_pot(finite, threshold_q)
    out = np.full(len(x), np.nan)
    for i, v in enumerate(losses):
        if not np.isfinite(v):
            continue
        p = tail_exceedance_probability(float(v), fit)
        # Normalize: esigin altinda 0; asiri kuyrukta p kucuk -> anomali buyuk.
        out[i] = float(np.clip(1.0 - p, 0.0, 1.0)) if v > fit.threshold else 0.0
    return pd.Series(out, index=log_ret.index, name="sig_evt")
