# SPDX-License-Identifier: AGPL-3.0-only
"""Agirlik kalibrasyonu.

Akademik dayanak:
  * Optimizasyon teorisi: kisitli agirlik dagilimi
        min_w  L(w)   s.t.  sum_j w_j = 1,  w_j >= 0
  * L(w) = || X w - y ||^2 (kisitli en kucuk kareler). Etiketli zemin gercegi
    bulunmadigindan y (hedef) sentetik senaryolardan veya uzman sezgisinden gelir;
    kalibrasyon bir "oneri" olarak sunulur, kesin dogru kabul edilmez.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..config import FusionWeights
from ..signals import SIGNAL_COLUMNS

_SIGNAL_TO_WEIGHT = {
    "sig_volume": "volume_z",
    "sig_index_divergence": "index_divergence",
    "sig_volatility": "volatility",
    "sig_mahalanobis": "mahalanobis",
    "sig_isolation_forest": "isolation_forest",
    "sig_liquidity": "liquidity",
    "sig_ml": "ml",
}


@dataclass
class CalibrationResult:
    weights: FusionWeights
    loss: float
    success: bool


def fit_weights(
    signals: np.ndarray,
    target: np.ndarray,
    *,
    l2: float = 1e-4,
    init: FusionWeights | None = None,
) -> CalibrationResult:
    """Kisitli agirliklari (sum=1, w>=0) hedefe gore kalibre eder.

    Args:
        signals: (n, k) normalize sinyal matrisi (SIGNAL_COLUMNS sirasiyla).
        target: (n,) hedef skor [0,1] (sentetik/uzman).
    """
    X = np.asarray(signals, dtype=float)
    y = np.asarray(target, dtype=float)
    if X.shape[1] != len(SIGNAL_COLUMNS):
        raise ValueError(f"Beklenen {len(SIGNAL_COLUMNS)} sinyal, gelen {X.shape[1]}")
    Xf = np.nan_to_num(X, nan=0.0)

    w0 = np.array([getattr(init, _SIGNAL_TO_WEIGHT[c]) for c in SIGNAL_COLUMNS]) if init else np.full(X.shape[1], 1.0 / X.shape[1])
    w0 = w0 / w0.sum()

    def objective(w: np.ndarray) -> float:
        resid = Xf @ w - y
        return float(resid @ resid / len(y) + l2 * (w @ w))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * X.shape[1]
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 500, "ftol": 1e-9})

    w = np.clip(res.x, 0.0, 1.0)
    w = w / w.sum() if w.sum() > 0 else w0
    kwargs = {_SIGNAL_TO_WEIGHT[c]: float(w[i]) for i, c in enumerate(SIGNAL_COLUMNS)}
    return CalibrationResult(weights=FusionWeights(**kwargs), loss=float(res.fun), success=bool(res.success))
