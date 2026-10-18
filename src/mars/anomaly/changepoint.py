# SPDX-License-Identifier: AGPL-3.0-only
"""Bayesian Online Change-Point Detection (BOCPD).

Akademik dayanak:
  * Adams, R. P., MacKay, D. J. C. (2007). Bayesian Online Changepoint Detection.
    arXiv:0710.3742.

Fikir: zaman serisinde rejim degisimlerini (kirilma noktalari) olasiliksal olarak
izler. Normal getiri rejiminden aniden sapma, supheli bir yapisal degisim olabilir.

Gaussian gozlem modeli (bilinen gozlem varyansi) + sabit changepoint onsel
olasiligi H. Her gun icin run-length = 0 sonrasi olasiligi (yani "bu gun kirilma
noktasi") sinyal olarak doner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Uzun run'lari budama (hesaplama maliyeti O(n * max_runs)).
_MAX_RUNS = 300


def bocpd_changepoint_prob(
    x: np.ndarray,
    hazard: float = 1.0 / 120.0,
    obs_var: float | None = None,
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
) -> np.ndarray:
    """Her gozlem icin changepoint (run-length=0) sonrasi olasiligi dondurur."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    cp = np.zeros(n)
    if n == 0:
        return cp
    if obs_var is None:
        # Gozlem varyansi icin robust olcek. Ham serinin MAD'i rejim kaymasiyla
        # bozulur; lag-1 farklarin MAD'i rejim-ici gurultuyu yansitir (kayma tek
        # buyuk atlama olarak gorunur ve medyani bozmaz).
        d = np.diff(x)
        d = d[np.isfinite(d)]
        if len(d) >= 5:
            mad = np.median(np.abs(d - np.median(d)))
            scale = (1.4826 * mad) / np.sqrt(2.0) if mad > 0 else (np.std(d) / np.sqrt(2.0) or 1e-4)
        else:
            med = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - med))
            scale = 1.4826 * mad if mad > 0 else (np.nanstd(x) or 1e-4)
        obs_var = float(scale ** 2) if scale > 0 else 1e-4

    # Run istatistikleri: her aktif run icin (sum, count)
    sums = np.array([0.0])
    counts = np.array([0.0])
    # R[r] = P(run length = r | x_1..t)
    R = np.array([1.0])

    for t in range(n):
        xt = x[t]
        if not np.isfinite(xt):
            cp[t] = 0.0
            continue

        # Standart Adams-MacKay ozyinelemesi. Changepoint dali (r_t=0) YENI rejimin
        # PRIOR prediktifini kullanir: pi_0(x_t) = N(xt | prior_mean, obs_var+prior_var).
        #   R_t(0)   = (1/Z) * H     * pi_0(xt) * sum_r R_{t-1,r}
        #   R_t(r+1) = (1/Z) * (1-H) * pi(xt | r) * R_{t-1,r}
        run_mean = np.where(counts > 0, sums / np.maximum(counts, 1.0), prior_mean)
        pred_var = obs_var * (1.0 + 1.0 / (counts + 1.0))
        pred = stats.norm.pdf(xt, loc=run_mean, scale=np.sqrt(pred_var))
        pred = np.clip(pred, 1e-300, None)

        cp_pred = float(stats.norm.pdf(xt, loc=prior_mean, scale=np.sqrt(obs_var + prior_var)))
        new_R = np.empty(len(R) + 1)
        new_R[0] = hazard * cp_pred * np.sum(R)
        new_R[1:] = (1.0 - hazard) * R * pred
        new_R = np.clip(new_R, 0.0, None)
        total = new_R.sum()
        new_R = new_R / total if total > 0 and np.isfinite(total) else np.ones_like(new_R) / len(new_R)

        cp[t] = float(new_R[0])

        # Run istatistiklerini guncelle
        new_sums = np.empty(len(sums) + 1)
        new_counts = np.empty(len(counts) + 1)
        new_sums[0] = 0.0
        new_counts[0] = 0.0
        new_sums[1:] = sums + xt
        new_counts[1:] = counts + 1.0

        if len(new_R) > _MAX_RUNS:
            new_R = new_R[-_MAX_RUNS:]
            new_sums = new_sums[-_MAX_RUNS:]
            new_counts = new_counts[-_MAX_RUNS:]

        R, sums, counts = new_R, new_sums, new_counts

    return cp


def changepoint_signal(log_ret: pd.Series, hazard: float = 1.0 / 120.0) -> pd.Series:
    prob = bocpd_changepoint_prob(log_ret.to_numpy(dtype=float), hazard=hazard)
    return pd.Series(prob, index=log_ret.index, name="sig_changepoint")
