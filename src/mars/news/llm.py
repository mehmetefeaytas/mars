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
