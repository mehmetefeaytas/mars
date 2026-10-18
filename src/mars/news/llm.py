# SPDX-License-Identifier: AGPL-3.0-only
"""Opsiyonel yerel/bulut LLM kopru katmani.

OpenAI-uyumlu bir endpoint (varsayilan Ollama: http://localhost:11434/v1) uzerinden
haber polaritesi/baglam analizi icin kisa bir JSON cikisi ister. VeriShield ile
ayni desen: LLM yoksa sistem kural-tabanli fallback ile calismaya devam eder.
"""

from __future__ import annotations

import json

import requests

from ..config import Settings, load_settings
from .sources import SentimentResult

_SYSTEM = (
    "Sen bir finansal haber analistisin. Verilen Turkce haber basligini/ozetini "
    "incele ve SADECE su JSON semasini dondur: "
    '{"polarity": <float -1..1>, "label": "pozitif|negatif|notr", "confidence": <float 0..1>}. '
    "Baska aciklama yazma."
)


def llm_available(settings: Settings | None = None) -> bool:
    settings = settings or load_settings()
    if not settings.llm_enabled:
        return False
    try:
        r = requests.get(f"{settings.llm_base_url.rstrip('/')}/models", timeout=2)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def llm_sentiment(text: str, settings: Settings | None = None) -> SentimentResult | None:
    """LLM ile polarite. Basarisizsa None (cagiran fallback yapar)."""
    settings = settings or load_settings()
    if not settings.llm_enabled:
        return None
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": text[:2000]},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        data = json.loads(content)
        return SentimentResult(
            polarity=float(data.get("polarity", 0.0)),
            label=str(data.get("label", "notr")),
            confidence=float(data.get("confidence", 0.5)),
            backend="llm",
        )
    except Exception:  # noqa: BLE001
        return None


_EXPLAIN_SYSTEM = (
    "Sen Borsa İstanbul izleme sistemi MARS'ın analist asistanısın. Sana verilen "
    "bir güne ait sinyal verilerini ve haber bilgilerini yorumlayacaksın. "
    "Kurallar: (1) Türkçe yaz. (2) En fazla 180 kelime. (3) Kesin manipülasyon "
    "hükmü verme; 'şüpheli/anormal' gibi ihtiyatlı dil kullan. (4) Yapı: kısa "
    "özet → hangi sinyaller skoru neden artırdı (sayılarla) → haber varsa "
    "açıklayıcı/çelişen rolü → sonuç ve izleme önerisi. (5) Yatırım tavsiyesi verme."
)


def build_explanation_prompt(payload: dict) -> str:
    """LLM'e verilecek kullanici istemini kurar (payload: gun verisi + haberler)."""
    lines = [
        f"Hisse: {payload.get('ticker')}",
        f"Tarih: {payload.get('date')}",
        f"Şüphe skoru: {payload.get('suspicion')}/100",
        f"Veri güveni: {payload.get('data_confidence')}/100",
        f"Günlük getiri: {payload.get('log_ret')}",
        f"Anormal getiri (AR): {payload.get('abnormal_ret')}",
        "Sinyal katkıları (puan):",
    ]
    for r in payload.get("reasons", []):
        lines.append(f"- {r.get('signal')}: +{r.get('points')} puan — {r.get('detail')}")
    news = payload.get("news", [])
    if news:
        lines.append("Eş zamanlı haberler:")
        for n in news:
            lines.append(
                f"- [{n.get('source')}] {n.get('headline')} "
                f"(polarite {n.get('polarity'):+.2f}, güvenilirlik {n.get('credibility'):.2f})"
            )
    else:
        lines.append("Eş zamanlı doğrulanmış haber bulunmuyor.")
    lines.append("Bu günü yukarıdaki kurallara göre açıkla.")
    return "\n".join(lines)


def explain_with_llm(payload: dict, settings: Settings | None = None) -> str | None:
    """Yerel LLM ile gunun aciklamasi. Basarisizsa None (kural-tabanli fallback)."""
    settings = settings or load_settings()
    if not settings.llm_enabled:
        return None
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    body = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": _EXPLAIN_SYSTEM},
            {"role": "user", "content": build_explanation_prompt(payload)},
        ],
        "temperature": 0.3,
        "max_tokens": 500,  # kacirma/uzun uretim riskini sinirlar
    }
    try:
        r = requests.post(url, json=body, timeout=settings.llm_timeout)
        r.raise_for_status()
        return str(r.json()["choices"][0]["message"]["content"]).strip()
    except Exception:  # noqa: BLE001
        return None
