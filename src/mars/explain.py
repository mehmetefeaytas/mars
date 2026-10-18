# SPDX-License-Identifier: AGPL-3.0-only
"""Aciklanabilirlik — sinyal katki ayristirmasi ve dogal dil ozeti.

Akademik dayanak:
  * Ribeiro, Singh, Guestrin (2016). LIME - yerel aciklanabilir temsiller.
  * Lundberg & Lee (2017). SHAP - katki degerleri.
  * Dogrusal füzyon yapisindan katki paylari (waterfall) dogrudan hesaplanir;
    iForest icin SHAP opsiyoneldir.
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


def _fmt_pct(x: float) -> str:
    return f"%{x * 100:+.1f}"


def _level(z: float) -> str:
    az = abs(z)
    if az < 1.5:
        return "normal"
    if az < 3:
        return "yüksek"
    if az < 5:
        return "aşırı"
    return "olağandışı"


def detailed_reasons(signal_row: pd.Series, fusion_row: pd.Series) -> list[dict]:
    """Her sinyal icin insan-okunur, DEGER BAZLI gerekce listesi uretir.

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
        add("Hacim sapması",
            f"İşlem hacmi, yaklaşık 60 günlük medyanına göre robust z = {vz:+.1f} "
            f"({abs(vz):.1f} MAD birim sapma, {_level(vz)} seviye).",
            pts("sig_volume"), _level(vz))

    # --- Oynaklik ---
    vlt = signal_row.get("volatility_z")
    if vlt is not None and np.isfinite(vlt):
        add("Oynaklık",
            f"Son gün getirilerin oynaklığı, yuvarlanan pencereye göre robust z = {vlt:+.1f} "
            f"({_level(vlt)} seviye).",
            pts("sig_volatility"), _level(vlt))

    # --- Likidite ---
    ilz = signal_row.get("illiquidity_z")
    if ilz is not None and np.isfinite(ilz):
        add("Likidite (Amihud)",
            f"Amihud illiquidity göstergesi robust z = {ilz:+.1f}; "
            + ("birim hacim başına fiyat etkisi artmış (sığ tabla işareti)." if ilz > 1.5
               else "olağan aralıkta."),
            pts("sig_liquidity"), _level(ilz))

    # --- Endeks ayrisimasi (market model) ---
    ar = signal_row.get("abnormal_ret")
    lr = signal_row.get("log_ret")
    mr = signal_row.get("market_log_ret")
    if ar is not None and np.isfinite(ar):
        parts = [f"Market modeline göre anormal getiri AR = {_fmt_pct(ar)}"]
        if lr is not None and np.isfinite(lr):
            mkt_txt = _fmt_pct(mr) if (mr is not None and np.isfinite(mr)) else "—"
            parts.append(f"(hisse {_fmt_pct(lr)}, endeks {mkt_txt})")
        parts.append(
            "— hareket endeks hareketiyle açıklanamıyor."
            if abs(ar) > 0.03 else "— hareket büyük ölçüde endeksle uyumlu."
        )
        add("Endeksten kopuş", " ".join(parts), pts("sig_index_divergence"),
            _level(ar / 0.02))

    # --- MCD Mahalanobis ---
    mh = signal_row.get("sig_mahalanobis")
    if mh is not None and np.isfinite(mh):
        detail = (
            f"5 boyutlu özellik uzayında robust Mahalanobis anomali olasılığı = {mh:.2f} "
            + ("(gözlem, normal piyasa davranış bulutunun belirgin dışında)."
               if mh > 0.8 else "(gözlem normal davranış bölgesinde).")
        )
        add("Çok değişkenli uzaklık (MCD)", detail,
            pts("sig_mahalanobis"), _level((mh - 0.5) * 6))

    # --- Isolation Forest ---
    ifo = signal_row.get("sig_isolation_forest")
    if ifo is not None and np.isfinite(ifo):
        detail = (
            f"Ortalama izolasyon derinliğine göre anomali skoru = {ifo:.2f} "
            + ("(normal gözlemlerden kısa yolda izole edildi)."
               if ifo > 0.8 else "(normal izolasyon derinliğinde).")
        )
        add("Isolation Forest", detail,
            pts("sig_isolation_forest"), _level((ifo - 0.5) * 6))

    # --- ML ---
    mlv = signal_row.get("sig_ml")
    if mlv is not None and np.isfinite(mlv):
        add("ML şüphe olasılığı",
            f"Eğitilmiş sınıflandırıcı şüphe olasılığı = {mlv:.2f} "
            + "(sentetik pump/lijit senaryolarından öğrenildi).",
            pts("sig_ml"), _level((mlv - 0.5) * 6))

    # --- Kapanis konumu (CLV) ---
    clv = signal_row.get("clv")
    if clv is not None and np.isfinite(clv):
        loc = "aralığın tepesinde" if clv > 0.5 else "aralığın dibinde" if clv < -0.5 else "aralık ortasında"
        add("Kapanış konumu (CLV)",
            f"Gün içi aralıkta kapanış {loc} (CLV = {clv:+.2f}); "
            + "kapanışa doğru fiyat itme (marking the close) izi."
            if abs(clv) > 0.7 else "olağan kapanış konumu.",
            pts("sig_close_location"), _level(abs(clv) * 4))

    # --- Cok gunlu desen ---
    car5 = signal_row.get("car5")
    streak = signal_row.get("up_streak")
    if car5 is not None and np.isfinite(car5):
        streak_txt = f", {int(streak)} günlük aynı-yön serisi" if streak is not None and np.isfinite(streak) else ""
        add("Çok günlük desen",
            f"Son 5 günün kümülatif anormal getirisi CAR₅ = {_fmt_pct(car5)}{streak_txt}; "
            + ("kalıcı, hacimle beslenen bir ayrışma deseni." if abs(car5) > 0.08
               else "kalıcı desen zayıf."),
            pts("sig_pump_pattern"), _level(abs(car5) / 0.04))

    # --- Tahta baglami ---
    tier = signal_row.get("liquidity_tier")
    prior = signal_row.get("tahta_prior")
    if tier is not None and tier != "" and prior is not None and np.isfinite(prior):
        add("Tahta bağlamı (likidite)",
            f"Hisse {tier} likidite katmanında (tahta riski önseli {prior:.2f}); "
            + ("hacim sinyali bu bağlamda aşırlıyor." if prior < 0.4
               else "hacim sinyali tam ağırlığında."),
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
    """Bir gun icin insan-okunur aciklama (zengin metin)."""
    suspicion = float(fusion_row.get("suspicion", 0.0))
    data_conf = float(fusion_row.get("data_confidence", 0.0))
    wf = waterfall(fusion_row, top_k=top_k)
    parts = []
    head = f"{date.date() if date is not None else ''}".strip()
    parts.append(f"{head} Şüphe skoru {suspicion:.0f}/100 (veri güveni {data_conf:.0f}/100).")

    band = ("düşük" if suspicion < 35 else "orta" if suspicion < 65 else "yüksek")
    parts.append(f"Risk bandı: {band}.")

    if wf:
        parts.append(
            "Öne çıkan sinyaller: "
            + ", ".join(f"{w['signal']} (+{w['points']})" for w in wf)
            + "."
        )
    if signal_row is not None:
        lr = signal_row.get("log_ret")
        if lr is not None and np.isfinite(lr):
            parts.append(f"Günlük getiri {_fmt_pct(lr)}.")

    factor = float(fusion_row.get("news_factor", 1.0))
    notes = fusion_row.get("news_notes", [])
    if factor < 0.999 and notes:
        parts.append("Dengeleyici faktör: " + " ".join(notes[:2]))
    elif factor > 1.001 and notes:
        parts.append("Artırıcı faktör: " + " ".join(notes[:2]))
    else:
        parts.append("Eş zamanlı doğrulanmış haber desteği bulunmuyor.")
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
