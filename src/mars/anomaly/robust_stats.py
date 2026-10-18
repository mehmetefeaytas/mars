# SPDX-License-Identifier: AGPL-3.0-only
"""Robust cok degiskenli istatistik.

Akademik dayanaklar:
  * Rousseeuw, P. J. (1984). Least Median of Squares Regression. JASA.
    -> Minimum Covariance Determinant (MCD) ile robust kovaryans tahmini.
  * Rousseeuw & Hubert (2011). Robust statistics for outlier detection. WIREs.
  * Mahalanobis uzakligi: d^2 = (x - mu)^T Sigma^{-1} (x - mu) ~ chi^2_m.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.covariance import MinCovDet


@dataclass
class MahalanobisResult:
    distances: np.ndarray      # robust Mahalanobis mesafesi (karekok)
    anomaly: np.ndarray        # [0,1] normalize anomali sinyali (yuksek=anormal)
    p_values: np.ndarray       # chi^2 tabanli ust kuyruk olasiligi


def robust_mahalanobis(X: np.ndarray, support_fraction: float | None = None) -> MahalanobisResult:
    """MCD ile robust Mahalanobis uzakligi ve normalize anomali sinyali.

    Args:
        X: (n, m) ozellik matrisi.
        support_fraction: MCD destek orani (None -> sklearn otomatik).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X iki boyutlu olmali (n x m).")
    n, m = X.shape
    if n < m + 2:
        # Yetersiz ornek: anomali sinyali uretemeyiz, notr dondur.
        zeros = np.zeros(n)
        return MahalanobisResult(zeros, zeros, np.ones(n))

    mcd = MinCovDet(support_fraction=support_fraction, random_state=42).fit(X)
    dist = mcd.mahalanobis(X)  # zaten kare uzaklik
    dist = np.clip(dist, 0.0, None)
    radius = np.sqrt(dist)

    # chi^2 ust kuyruk: kucuk p -> anormal. Anomali = 1 - p.
    p_values = stats.chi2.sf(dist, df=m)
    anomaly = 1.0 - p_values
    return MahalanobisResult(radius, anomaly, p_values)


def robust_zscore(x: np.ndarray, scale: float = 1.4826) -> np.ndarray:
    """Tek degiskenli robust z (medyan/MAD)."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    denom = scale * mad
    if denom == 0:
        sd = np.std(x, ddof=0)
        denom = sd if sd > 0 else 1.0
    return (x - med) / denom


def squash_to_unit(z: np.ndarray, k: float = 1.0) -> np.ndarray:
    """Robust z'yi [0,1] araligina sigmoid benzeri sikistirir (yuksek=anormal)."""
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-k * (z - 3.0)))
