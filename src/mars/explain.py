# SPDX-License-Identifier: AGPL-3.0-only
"""Aciklanabilirlik — sinyal katki ayristirmasi ve sade dil ozeti.

Bu modulun urettigi metinler bilincli olarak BASIT TUTULMUSTUR: okuyucunun
istatistik bilgisi olmadigi varsayilir. Teknik terimler parantez icinde bir
cumleyle aciklanir.

Akademik dayanak:
  * Ribeiro, Singh, Guestrin (2016). LIME - yerel aciklanabilir temsiller.
  * Lundberg & Lee (2017). SHAP - katki degerleri.
  * Dogrusal füzyon yapisindan katki paylari (waterfall) dogrudan hesaplanir.
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
    """Katki şelalesi: skoru en cok artiran sinyaller (puan)."""
    df = contributions_row(fusion_row).head(top_k)
    return [{"signal": r.signal, "points": round(r.points, 1)} for r in df.itertuples()]


def _fmt_pct(x: float) -> str:
    return f"%{x * 100:+.1f}"


def _level(z: float) -> str:
    """Sapma buyuklugunu sade kelimeyle ifade eder."""
    az = abs(z)
    if az < 1.5:
        return "normal"
    if az < 3:
        return "biraz yüksek"
    if az < 5:
        return "yüksek"
    return "çok yüksek"


def detailed_reasons(signal_row: pd.Series, fusion_row: pd.Series) -> list[dict]:
    """Her sinyal icin sade dille, SAYIYA DAYALI gerekce listesi uretir.

    Args:
        signal_row: compute_signals ciktisindan bir gun (ham ozellikler + sinyaller).
        fusion_row: fuse ciktisindan ayni gun (katkilar + skor).
    """
    reasons: list[dict] = []

    def add(label: str, detail: str, points: float, level: str):
        reasons.append({"signal": label, "detail": detail, "points": round(points, 1), "level": level})

    def pts(col: str) -> float:
        return float(fusion_row.get(f"contrib_{col}", 0.0) or 0.0)

    # --- Hacim ---
    vz = signal_row.get("vol_z")
    if vz is not None and np.isfinite(vz):
        yon = "üstünde" if vz > 0 else "altında"
        add("İşlem hacmi",
            f"Bugünkü alım-satım hacmi, son ~60 günün normal seviyesinin {abs(vz):.1f} adım "
            f"{yon}. Bu sapmanın büyüklüğü: {_level(vz)}.",
            pts("sig_volume"), _level(vz))

    # --- Oynaklik ---
    vlt = signal_row.get("volatility_z")
    if vlt is not None and np.isfinite(vlt):
        add("Fiyat oynaklığı",
            f"Fiyatın gün içinde ne kadar sıçradığı (oynaklık), son dönemin normaline göre "
            f"{_level(vlt)} seviyede.",
            pts("sig_volatility"), _level(vlt))

    # --- Likidite ---
    ilz = signal_row.get("illiquidity_z")
    if ilz is not None and np.isfinite(ilz):
        add("Likidite (alım-satım kolaylığı)",
            "Hisse sığ: küçük bir alım-satım bile fiyatı normalden fazla oynatıyor."
            if ilz > 1.5 else "Alım-satım kolaylığı olağan seviyede.",
            pts("sig_liquidity"), _level(ilz))

    # --- Endeks ayrisimasi (market model) ---
    ar = signal_row.get("abnormal_ret")
    lr = signal_row.get("log_ret")
    mr = signal_row.get("market_log_ret")
    if ar is not None and np.isfinite(ar):
        parts = [f"Hisse, genel piyasanın (endeks) hareketiyle açıklanamayan {_fmt_pct(ar)} "
                 "fazladan getiri üretti"]
        if lr is not None and np.isfinite(lr):
            mkt_txt = _fmt_pct(mr) if (mr is not None and np.isfinite(mr)) else "—"
            parts.append(f"(hisse {_fmt_pct(lr)}, endeks {mkt_txt})")
        parts.append(
            "— hareket endeksten kopmuş." if abs(ar) > 0.03
            else "— hareket büyük ölçüde endeksle uyumlu."
        )
        add("Endeksten kopma", " ".join(parts), pts("sig_index_divergence"),
            _level(ar / 0.02))

    # --- MCD Mahalanobis ---
    mh = signal_row.get("sig_mahalanobis")
    if mh is not None and np.isfinite(mh):
        yorum = ("normalin belirgin şekilde dışında." if mh > 0.8 else "normal aralıkta.")
        detail = (
            "Bugünkü hareket, hissenin normal davranış kalıbına ne kadar uymuyor? "
            f"Uymama puanı {mh:.2f} ({yorum})"
        )
        add("Genel anormallik (MCD)", detail,
            pts("sig_mahalanobis"), _level((mh - 0.5) * 6))

    # --- Isolation Forest ---
    ifo = signal_row.get("sig_isolation_forest")
    if ifo is not None and np.isfinite(ifo):
        yorum = ("normal günlerden kolayca ayırt ediliyor." if ifo > 0.8
                 else "normal günlere benziyor.")
        detail = (
            "Hareket normal günlere ne kadar benzemiyor? "
            f"Benzersizlik puanı {ifo:.2f} ({yorum})"
        )
        add("Benzersiz hareket (iForest)", detail,
            pts("sig_isolation_forest"), _level((ifo - 0.5) * 6))

    # --- ML ---
    mlv = signal_row.get("sig_ml")
    if mlv is not None and np.isfinite(mlv):
        add("Öğrenilmiş şüphe (ML)",
            f"Verilerle eğitilen modelin bu günü 'şüpheli' görme olasılığı {mlv:.2f} "
            "(1'e yakın = daha şüpheli).",
            pts("sig_ml"), _level((mlv - 0.5) * 6))

    # --- Kapanis konumu (CLV) ---
    clv = signal_row.get("clv")
    if clv is not None and np.isfinite(clv):
        loc = "günün en yüksek seviyelerine yakın" if clv > 0.5 else \
              "günün en düşük seviyelerine yakın" if clv < -0.5 else "günün ortasında"
        extra = (" Kapanışa doğru fiyatı yukarı itme izi var (alıcı baskısı)."
                 if abs(clv) > 0.7 else " Kapanış konumu olağan.")
        add("Kapanış konumu",
            f"Gün kapanışı {loc} (kapanış gücü {clv:+.2f})." + extra,
            pts("sig_close_location"), _level(abs(clv) * 4))

    # --- Cok gunlu desen ---
    car5 = signal_row.get("car5")
    streak = signal_row.get("up_streak")
    if car5 is not None and np.isfinite(car5):
        yon = "yukarı" if car5 > 0 else "aşağı"
        streak_txt = (f", üst üste {int(streak)} gün aynı yönde" 
                      if streak is not None and np.isfinite(streak) and streak >= 2 else "")
        add("Çok günlük desen",
            f"Son 5 günün toplam anormal getirisi {_fmt_pct(car5)} ({yon}{streak_txt}). "
            + ("Birkaç gündür süren, hacimle beslenen bir hareket var."
               if abs(car5) > 0.08 else "Sürekli bir desen zayıf."),
            pts("sig_pump_pattern"), _level(abs(car5) / 0.04))

    # --- Tahta baglami ---
    tier = signal_row.get("liquidity_tier")
    prior = signal_row.get("tahta_prior")
    if tier is not None and tier != "" and prior is not None and np.isfinite(prior):
        add("Tahta bağlamı (likidite)",
            f"Bu hisse '{tier}' likidite grubunda. "
            + ("Çok alıcılı/satıcılı bir tahta olduğu için hacim sinyali kısılıyor; "
               "hareket daha büyük olmalı ki dikkat çeksin."
               if prior < 0.4 else "Hacim sinyali tam ağırlıkta değerlendiriliyor."),
            0.0, "bağlam")

    reasons.sort(key=lambda r: -r["points"])
    return reasons


def summarize_day(
    fusion_row: pd.Series,
    *,
    date: pd.Timestamp | None = None,
    signal_row: pd.Series | None = None,
    top_k: int = 3,
) -> str:
    """Bir gun icin sade dilde ozet (otomatik uretilir, LLM'siz de calisir)."""
    suspicion = float(fusion_row.get("suspicion", 0.0))
    data_conf = float(fusion_row.get("data_confidence", 0.0))
    wf = waterfall(fusion_row, top_k=top_k)
    parts = []
    head = f"{date.date() if date is not None else ''}".strip()
    parts.append(f"{head} için şüphe skoru {suspicion:.0f}/100 (veri güveni {data_conf:.0f}/100).")

    if suspicion < 35:
        band = "Düşük: bugün sıra dışı bir şey görünmüyor."
    elif suspicion < 65:
        band = "Orta: dikkat çeken, ama kesin olmayan işaretler var."
    else:
        band = "Yüksek: birden çok işaret aynı yönde."
    parts.append(band)

    if wf:
        parts.append(
            "Skoru en çok artıranlar: "
            + ", ".join(f"{w['signal']} (+{w['points']} puan)" for w in wf)
            + "."
        )
    if signal_row is not None:
        lr = signal_row.get("log_ret")
        if lr is not None and np.isfinite(lr):
            parts.append(f"Günlük getiri {_fmt_pct(lr)}.")

    factor = float(fusion_row.get("news_factor", 1.0))
    notes = fusion_row.get("news_notes", [])
    if factor < 0.999 and notes:
        parts.append("Haber skoru düşürdü: " + " ".join(notes[:2]))
    elif factor > 1.001 and notes:
        parts.append("Haber skoru artırdı: " + " ".join(notes[:2]))
    else:
        parts.append("Skoru açıklayan doğrulanmış bir haber bulunmuyor.")
    return " ".join(parts)


def shap_iforest_contributions(model, X: np.ndarray, index: int) -> dict[int, float] | None:
    """Opsiyonel: iForest icin SHAP degerleri (shap kuruluysa)."""
    try:
        import shap  # noqa: PLC0415

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
        return {i: float(v) for i, v in enumerate(values[index])}
    except Exception:  # noqa: BLE001
        return None
