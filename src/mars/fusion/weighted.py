# SPDX-License-Identifier: AGPL-3.0-only
"""Kisitli agirlikli füzyon + haber duzeltmesi.

Denklem (rapor Eq. 5):
    S_t = 100 * ( sum_j w_j A_j,t ) / ( sum_j w_j )
burada A_j,t in [0,1] normalize sinyaller, w_j >= 0 agirliklar.

Haber duzeltmesi (Bayesci sezgi): dogrulanmis ve hareketle AYNI yonde bir haber
supheyi dusurur; CELISEN guvenilir haber supheyi hafifce artirir. Etki,
`news_penalty_strength` ile sinirlanir (tek basina veto degildir).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import FusionWeights, Settings, load_settings
from ..news.sources import NewsSignal
from ..signals import SIGNAL_COLUMNS

# Sinyal adi -> füzyon agirlik anahtari eslemesi
_SIGNAL_TO_WEIGHT = {
    "sig_volume": "volume_z",
    "sig_index_divergence": "index_divergence",
    "sig_volatility": "volatility",
    "sig_mahalanobis": "mahalanobis",
    "sig_isolation_forest": "isolation_forest",
    "sig_liquidity": "liquidity",
}

_SIGNAL_LABELS = {
    "sig_volume": "Hacim sapmasi",
    "sig_index_divergence": "Endeksten kopuş",
    "sig_volatility": "Oynaklik patlamasi",
    "sig_mahalanobis": "Cok degiskenli uzaklik",
    "sig_isolation_forest": "Isolation Forest",
    "sig_liquidity": "Likidite soku",
}


@dataclass
class FusionResult:
    frame: pd.DataFrame
    weights: dict[str, float]
    signal_labels: dict[str, str] = field(default_factory=lambda: dict(_SIGNAL_LABELS))


def _normalize_weights(weights: FusionWeights | dict[str, float]) -> dict[str, float]:
    if isinstance(weights, FusionWeights):
        return weights.normalized()
    total = sum(weights.values())
    if total <= 0:
        n = max(1, len(weights))
        return {k: 1.0 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


def _data_confidence(row: pd.Series, n_history: int, feature_count: int) -> float:
    """Veri kapsami/derinligi skoru (0-100)."""
    available = row[SIGNAL_COLUMNS].notna().sum()
    completeness = available / max(1, len(SIGNAL_COLUMNS))
    depth = min(1.0, n_history / 250.0)
    return round(100.0 * (0.6 * completeness + 0.4 * depth), 1)


def _news_adjustment(
    ar: float,
    news: list[NewsSignal],
    strength: float,
) -> tuple[float, float, list[str]]:
    """Bir gun icin haber duzeltme faktoru.

    Returns: (factor, explained, reasons)
      factor: supheye carpilan carpan (1.0 = etki yok, <1 dusurur, >1 artirir)
    """
    if not news:
        return 1.0, 0.0, []

    confirmed = 0.0
    contradiction = 0.0
    reasons: list[str] = []
    ar_sign = 0.0 if abs(ar) < 1e-9 else np.sign(ar)

    for n in news:
        p = n.sentiment.polarity
        if abs(p) < 0.05:
            continue
        weight = n.credibility.score * min(1.0, abs(p))
        if ar_sign == 0 or np.sign(p) == ar_sign:
            confirmed += weight
            reasons.append(
                f"Dogrulanmis haber hareketle ayni yonde (guvenilirlik {n.credibility.score:.2f})."
            )
        else:
            contradiction += weight
            reasons.append(
                f"Guvenilir haber hareketle celisiyor (guvenilirlik {n.credibility.score:.2f})."
            )

    confirmed = min(1.0, confirmed)
    contradiction = min(1.0, contradiction)
    factor = 1.0 - strength * confirmed + 0.15 * contradiction
    factor = float(np.clip(factor, 0.15, 1.25))
    return factor, confirmed, reasons


def fuse(
    signal_frame: pd.DataFrame,
    *,
    news_by_date: dict[pd.Timestamp, list[NewsSignal]] | None = None,
    abnormal_returns: pd.Series | None = None,
    settings: Settings | None = None,
    weights: FusionWeights | dict[str, float] | None = None,
) -> FusionResult:
    """Sinyal cercevesini gunluk suphe skoruna (0-100) donusturur."""
    settings = settings or load_settings()
    weights = _normalize_weights(weights or settings.weights)
    w = np.array([weights[_SIGNAL_TO_WEIGHT[c]] for c in SIGNAL_COLUMNS])
    w_sum = w.sum() if w.sum() > 0 else 1.0

    signals = signal_frame[SIGNAL_COLUMNS]
    # NaN sinyaller 0 katki verir (bilinmeyen anormal sayilmaz).
    A = signals.to_numpy(dtype=float)
    A_filled = np.nan_to_num(A, nan=0.0)
    weighted = A_filled * w  # (n, k)
    suspicion = 100.0 * weighted.sum(axis=1) / w_sum

    n = len(signal_frame)
    result = pd.DataFrame(index=signal_frame.index)
    result["suspicion_base"] = suspicion

    # Katki paylari (puan cinsinden)
    for i, col in enumerate(SIGNAL_COLUMNS):
        result[f"contrib_{col}"] = 100.0 * weighted[:, i] / w_sum

    # Data confidence
    result["data_confidence"] = [
        _data_confidence(signal_frame.iloc[i], i + 1, len(SIGNAL_COLUMNS)) for i in range(n)
    ]

    # Haber duzeltmesi
    news_factor = np.ones(n)
    news_explained = np.zeros(n)
    news_notes: list[list[str]] = [[] for _ in range(n)]
    if news_by_date and abnormal_returns is not None:
        ar = abnormal_returns.reindex(signal_frame.index)
        for i, date in enumerate(signal_frame.index):
            items = news_by_date.get(date, [])
            f, expl, reasons = _news_adjustment(
                float(ar.iloc[i]) if np.isfinite(ar.iloc[i]) else 0.0,
                items,
                settings.news_penalty_strength,
            )
            news_factor[i] = f
            news_explained[i] = expl
            news_notes[i] = reasons

    result["news_factor"] = news_factor
    result["news_explained"] = news_explained
    result["news_notes"] = news_notes
    result["suspicion"] = np.clip(suspicion * news_factor, 0.0, 100.0)

    return FusionResult(frame=result, weights=weights)
