# SPDX-License-Identifier: AGPL-3.0-only
"""Kanit füzyonu — Dempster-Shafer (alternatif füzyon).

Akademik dayanak:
  * Shafer, G. (1976). A Mathematical Theory of Evidence. Princeton University Press.

Basit füzyon (weighted.py) agirlikli ortalamadir. Ancak her sinyalin guvenilirligi
farklidir; DS, her kanalin gorusunu bir inanç kütlesi (basic belief assignment, BBA)
olarak temsil edip celiski (K) durumunu acik bir sekilde ele almamiza izin verir.

Frame of discernment: {A (anomali), N (normal)}. U = {A, N} (cehalet).
Sinyal j icin:
    m_j({A})   = A_j,t * r_j
    m_j({N})   = (1 - A_j,t) * r_j
    m_j({A,N}) = 1 - r_j
Burada r_j in [0,1] o sinyalin guvenilirligi/agirlik normudur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..signals import SIGNAL_COLUMNS


def combine_two(m1: dict[str, float], m2: dict[str, float]) -> dict[str, float]:
    """Iki BBA'yi Dempster kuraliyla birlestirir. Anahtarlar: 'A','N','U'."""
    keys = ("A", "N", "U")
    conflict = m1["A"] * m2["N"] + m1["N"] * m2["A"]
    out = {k: 0.0 for k in keys}
    for x in keys:
        for y in keys:
            # Kesisim: U notr kume; A∩A=A, N∩N=N, A∩N=∅
            if x == "U":
                target = y
            elif y == "U":
                target = x
            elif x == y:
                target = x
            else:
                continue  # celiski -> K'ya gider
            out[target] += m1[x] * m2[y]
    if conflict >= 1.0:
        # Tam celiski: normalizasyon tanimsiz -> notr (0.5) kabul et.
        return {"A": 0.5, "N": 0.5, "U": 0.0}
    norm = 1.0 - conflict
    return {k: v / norm for k, v in out.items()}


def ds_belief_anomaly(
    masses: list[dict[str, float]],
) -> float:
    """Bir liste BBA'nin bilesik Bel(A) degeri (0..1)."""
    if not masses:
        return 0.5
    acc = masses[0]
    for m in masses[1:]:
        acc = combine_two(acc, m)
    return float(np.clip(acc["A"], 0.0, 1.0))


def signal_to_bba(a: float, r: float) -> dict[str, float]:
    a = float(np.clip(a, 0.0, 1.0))
    r = float(np.clip(r, 0.0, 1.0))
    return {"A": a * r, "N": (1.0 - a) * r, "U": 1.0 - r}


def fuse_evidence(
    signal_frame: pd.DataFrame,
    *,
    reliabilities: dict[str, float] | None = None,
) -> pd.Series:
    """Her gun icin DS bilesik anomali inanci (0-100 skalasina cevrilir)."""
    rel = reliabilities or {c: 1.0 for c in SIGNAL_COLUMNS}
    values = signal_frame[SIGNAL_COLUMNS].to_numpy(dtype=float)
    out = np.full(len(signal_frame), np.nan)
    for i in range(len(signal_frame)):
        masses = []
        for j, col in enumerate(SIGNAL_COLUMNS):
            a = values[i, j]
            if not np.isfinite(a):
                continue
            masses.append(signal_to_bba(a, rel.get(col, 1.0)))
        if masses:
            out[i] = 100.0 * ds_belief_anomaly(masses)
    return pd.Series(out, index=signal_frame.index, name="suspicion_ds")
