# SPDX-License-Identifier: AGPL-3.0-only
"""Tahta riski baglami — likidite katmanina gore manipulasyon yapilabilirligi onseli.

Akademik dayanak:
  * Aggarwal, R. K., Wu, G. (2006). Stock Market Manipulation — Theory and
    Evidence. Journal of Business 79(4): manipulasyon, dusuk likiditeli, dusuk
    fiyati, yuksek oynakliklu hisselerde yogunlasir.
  * Khwaja, A. I., Mian, A. (2005). Unchecked Intermediaries: Quantity
    Manipulation in an Emerging Stock Market. QJE 120(2): Pakistan KSE'de
    broker gucleri sig tahtalarda fiyat manipule eder.

Buna gore "cok hacimli/derinlikli" bir hissede (or. BIM) hacim sicramasi tek
basina tahtacilik kaniti degildir; hacim sinyalinin agirligi likidite baglamina
gore asirilir. Bu bir VETO degil, sinirli bir modulasyondur: derin tahtada bile
diger sinyaller (endeks ayrisimasi, CLV, desen) skoru yukseltebilir.
"""

from __future__ import annotations

import numpy as np

# (log10 medyan dolar hacim [TL], tahta_prior) ankrajlari — BIST gunluk TL hacimleri
# 20M TL alti "cok sig", 10B TL ustu "mega likit" kabul edilir.
_ANCHORS: list[tuple[float, float]] = [
    (7.0, 1.00),    # ~10M TL  — cok sig tahta: tahtacilik cok kolay
    (7.7, 0.90),    # ~50M TL
    (8.3, 0.70),    # ~200M TL
    (8.9, 0.50),    # ~800M TL
    (9.5, 0.32),    # ~3B TL
    (10.0, 0.15),   # ~10B TL — mega likit: tahtacilik cok zor
]

_TIERS = [
    (0.90, "çok sığ"),
    (0.60, "sığ"),
    (0.40, "orta"),
    (0.22, "likit"),
    (-1.0, "çok likit"),
]


def tahta_prior(adv_dollar_median: float) -> float:
    """Medyan gunluk dolar hacme gore tahta riski onseli (0..1).

    Log10 uzayinda dogrusal interpolasyon; ankrajlarin disinda kirpilir.
    """
    if not np.isfinite(adv_dollar_median) or adv_dollar_median <= 0:
        return 0.5  # bilinmiyor -> notr
    x = float(np.log10(adv_dollar_median))
    if x <= _ANCHORS[0][0]:
        return _ANCHORS[0][1]
    if x >= _ANCHORS[-1][0]:
        return _ANCHORS[-1][1]
    for (x0, p0), (x1, p1) in zip(_ANCHORS, _ANCHORS[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0)
            return float(p0 + t * (p1 - p0))
    return 0.5


def liquidity_tier(prior: float) -> str:
    """Tahta onselinden okunabilir likidite katmani etiketi."""
    for threshold, label in _TIERS:
        if prior >= threshold:
            return label
    return _TIERS[-1][1]
