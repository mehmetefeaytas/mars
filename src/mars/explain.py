# SPDX-License-Identifier: AGPL-3.0-only
"""Aciklanabilirlik — sinyal katki ayristirmasi ve dogal dil ozeti.

Akademik dayanak:
  * Ribeiro, Singh, Guestrin (2016). LIME - yerel aciklanabilir temsiller.
  * Lundberg & Lee (2017). SHAP - katki degerleri.
  * Burada dogrudan, modelin dogrusal füzyon yapisi kullanilarak katki paylari
    (waterfall) hesaplanir; ayrica iForest icin SHAP opsiyoneldir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fusion.weighted import _SIGNAL_LABELS

_REVERSE_LABELS = {v: k for k, v in _SIGNAL_LABELS.items()}


def contributions_row(fusion_row: pd.Series) -> pd.DataFrame:
    """Bir gunun sinyal katkilarini puan cinsinden tablo dondurur."""
    rows = []
    for col, label in _SIGNAL_LABELS.items():
        key = f"contrib_{col}"
        if key in fusion_row.index:
            rows.append({"signal": label, "points": float(fusion_row[key])})
    df = pd.DataFrame(rows).sort_values("points", ascending=False).reset_index(drop=True)
    return df


def waterfall(fusion_row: pd.Series, top_k: int = 5) -> list[dict]:
    """Katki şelalesi: en etkili sinyaller (puan)."""
    df = contributions_row(fusion_row).head(top_k)
    return [{"signal": r.signal, "points": round(r.points, 1)} for r in df.itertuples()]


def summarize_day(
    fusion_row: pd.Series,
    *,
    date: pd.Timestamp | None = None,
    top_k: int = 3,
) -> str:
    """Bir gun icin insan-okunur aciklama."""
    suspicion = float(fusion_row.get("suspicion", 0.0))
    data_conf = float(fusion_row.get("data_confidence", 0.0))
    wf = waterfall(fusion_row, top_k=top_k)
    parts = []
    head = f"{date.date() if date is not None else ''}".strip()
    parts.append(f"{head} Suphe skoru {suspicion:.0f}/100 (veri guveni {data_conf:.0f}/100).")
    if wf:
        parts.append(
            "One cikan sinyaller: "
            + ", ".join(f"{w['signal']} (+{w['points']})" for w in wf)
            + "."
        )
    factor = float(fusion_row.get("news_factor", 1.0))
    notes = fusion_row.get("news_notes", [])
    if factor < 0.999 and notes:
        parts.append("Dengeleyici faktor: " + " ".join(notes[:2]))
    elif factor > 1.001 and notes:
        parts.append("Artirici faktor: " + " ".join(notes[:2]))
    else:
        parts.append("Es zamanli dogrulanmis haber destegi bulunmuyor.")
    return " ".join(parts)


def shap_iforest_contributions(model, X: np.ndarray, index: int) -> dict[int, float] | None:
    """Opsiyonel: iForest icin SHAP degerleri (shap kuruluysa).

    shap agir bir bagimliliktir; yoksa None dondurur ve sistem dogrusal katki
    paylarina dayanmaya devam eder.
    """
    try:
        import shap  # noqa: PLC0415

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
        return {i: float(v) for i, v in enumerate(values[index])}
    except Exception:  # noqa: BLE001
        return None
