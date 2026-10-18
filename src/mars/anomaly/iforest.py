# SPDX-License-Identifier: AGPL-3.0-only
"""Isolation Forest anomali tespiti.

Akademik dayanak:
  * Liu, F. T., Ting, K. M., Zhou, Z. H. (2008). Isolation Forest. IEEE ICDM.
    s(x, n) = 2^{-E(h(x)) / c(n)}
    c(n) = 2 ln(n-1) + 0.5772156649 - 2(n-1)/n
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class IForestResult:
    score: np.ndarray       # [0,1] anomali sinyali (yuksek=anormal)
    raw: np.ndarray         # sklearn decision_function (dusuk=anormal)
    model: IsolationForest


def fit_isolation_forest(
    X: np.ndarray,
    n_estimators: int = 200,
    contamination: float = 0.02,
    random_state: int = 42,
) -> IForestResult:
    """Isolation Forest egitir ve [0,1] normalize anomali sinyali dondurur."""
    X = np.asarray(X, dtype=float)
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    # score_samples: yuksek -> normal. Once [0,1]'e normalize et, sonra ters cevir.
    raw = model.score_samples(X)
    lo, hi = float(np.min(raw)), float(np.max(raw))
    if hi - lo < 1e-12:
        normalized = np.full_like(raw, 0.5)
    else:
        normalized = (raw - lo) / (hi - lo)
    anomaly = 1.0 - normalized  # yuksek = anormal
    return IForestResult(score=anomaly, raw=raw, model=model)
