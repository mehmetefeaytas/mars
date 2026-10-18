# SPDX-License-Identifier: AGPL-3.0-only
"""Kisitli agirlikli füzyon + haber duzeltmesi.

Denklem (rapor Eq. 5):
    S_t = 100 * ( sum_j w_j A_j,t ) / ( sum_j w_j )
burada A_j,t in [0,1] normalize sinyaller, w_j >= 0 agirliklar.

Gun-bazinda normalize etme: bir sinyal o gun veri icermiyorsa (NaN — or. ML
modeli yok) agirligi otomatik olarak diger mevcut sinyallere yeniden dagitilir.
Boylece eksik sinyal skoru sistematik olarak dusurmez.

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
    "sig_ml": "ml",
    "sig_close_location": "close_location",
    "sig_pump_pattern": "pump_pattern",
}

_SIGNAL_LABELS = {
    "sig_volume": "İşlem hacmi",
    "sig_index_divergence": "Endeksten kopma",
    "sig_volatility": "Fiyat oynaklığı",
    "sig_mahalanobis": "Genel anormallik (MCD)",
    "sig_isolation_forest": "Benzersiz hareket (iForest)",
    "sig_liquidity": "Likidite",
    "sig_ml": "Öğrenilmiş şüphe (ML)",
    "sig_close_location": "Kapanış konumu",
    "sig_pump_pattern": "Çok günlük desen",
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


def _data_confidence(row: pd.Series, n_history: int, signal_cols: list[str]) -> float:
    """Veri kapsami/derinligi skoru (0-100)."""
    present = [c for c in signal_cols if c in row.index]
    available = row[present].notna().sum()
    completeness = available / max(1, len(present))
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
                f"Doğrulanmış haber hareketle aynı yönde (güvenilirlik {n.credibility.score:.2f})."
            )
        else:
            contradiction += weight
            reasons.append(
                f"Güvenilir haber hareketle çelişiyor (güvenilirlik {n.credibility.score:.2f})."
            )

    confirmed = min(1.0, confirmed)
    contradiction = min(1.0, contradiction)
    factor = 1.0 - strength * confirmed + 0.15 * contradiction
    factor = float(np.clip(factor, 0.15, 1.25))
    return factor, confirmed, reasons


def _apply_tahta_modulation(
    w: np.ndarray, cols: list[str], signal_frame: pd.DataFrame, settings: Settings
) -> np.ndarray:
    """Likidite baglamina gore hacim/likidite agirliklarini asirir.

    Aggarwal & Wu (2006) / Khwaja & Mian (2005): tahtacilik sig tahtalarda
    kolaylasir. Derin/mega likit bir hissede (dusuk tahta_prior) hacim
    sicramasinin agirligi azaltilir — VETO degil, sinirli modulasyon.
    Kalan agirlik gun-bazli normalizasyonla mevcut sinyallere dagitilir.
    """
    m = float(np.clip(settings.tahta_modulation, 0.0, 1.0))
    if m <= 0:
        return w
    if "tahta_prior" not in signal_frame.columns:
        return w
    prior = pd.to_numeric(signal_frame["tahta_prior"], errors="coerce").dropna()
    if prior.empty:
        return w
    p = float(prior.iloc[-1])

    w = w.copy()
    for i, col in enumerate(cols):
        if col == "sig_volume":
            w[i] *= 1.0 - m * 0.65 * (1.0 - p)
        elif col == "sig_liquidity":
            w[i] *= 1.0 - m * 0.50 * (1.0 - p)
    return w


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

    cols = [c for c in SIGNAL_COLUMNS if c in signal_frame.columns]
    w = np.array([weights.get(_SIGNAL_TO_WEIGHT[c], 0.0) for c in cols])
    w = _apply_tahta_modulation(w, cols, signal_frame, settings)

    signals = signal_frame[cols]
    A = signals.to_numpy(dtype=float)
    finite = np.isfinite(A)
    A0 = np.where(finite, A, 0.0)

    # Gun-bazinda agirlik normalizasyonu: yalniz mevcut (finite) sinyaller.
    w_day = (finite * w).sum(axis=1)                       # (n,)
    w_day_safe = np.where(w_day > 0, w_day, 1.0)
    weighted = A0 * w                                      # (n, k)
    suspicion = 100.0 * weighted.sum(axis=1) / w_day_safe
    suspicion = np.where(w_day > 0, suspicion, 0.0)

    n = len(signal_frame)
    result = pd.DataFrame(index=signal_frame.index)
    result["suspicion_base"] = suspicion

    # Katki paylari (puan cinsinden)
    for i, col in enumerate(cols):
        result[f"contrib_{col}"] = 100.0 * weighted[:, i] / w_day_safe

    # Data confidence
    result["data_confidence"] = [
        _data_confidence(signal_frame.iloc[i], i + 1, cols) for i in range(n)
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
