# SPDX-License-Identifier: AGPL-3.0-only
"""Calisma zamani ayar deposu (Ayarlar sayfasi icin).

Ortam degiskenleri (`.env`) salt-okunur kabul edilir; bu modul kullanicinin
arayuzden degistirdigi ayarlari `data/settings.local.json` dosyasinda tutar ve
`load_settings()` bu dosyayi env'in UZERINE uygular.

Sir (API anahtari) bu dosyada tutulur; `settings.local.json` .gitignore icindedir.
API yanitlarinda anahtar maskelenir.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Duzenlenebilir alanlar -> tip donusturucu
EDITABLE: dict[str, type] = {
    # LLM
    "llm_enabled": bool,
    "llm_provider": str,
    "llm_base_url": str,
    "llm_model": str,
    "llm_api_key": str,
    "llm_timeout": int,
    "llm_thinking": bool,
    "llm_max_tokens": int,
    "llm_temperature": float,
    # Haber
    "rss_enabled": bool,
    "rss_feeds": str,
    "rss_max_items": int,
    "sentiment_backend": str,
    "verishield_enabled": bool,
    # Analiz
    "lookback_days": int,
    "rolling_window": int,
    "estimation_window": int,
    "iforest_trees": int,
    "iforest_contamination": float,
    "news_penalty_strength": float,
    "tahta_modulation": float,
}

# Istemcide gosterilen sir alanlari
SECRET_FIELDS = {"llm_api_key"}

DEFAULT_PROVIDER_URLS = {
    "ollama": "http://localhost:11434/v1",
    "evren": "https://evren-llmapi.ssyz.org.tr/v1",
    "openai": "https://api.openai.com/v1",
}


def store_path() -> Path:
    base = Path(os.environ.get("MARS_DATA_DIR", "data"))
    return base / "settings.local.json"


def load_overrides(path: Path | None = None) -> dict[str, Any]:
    p = path or store_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in data.items():
        if k in EDITABLE and v is not None:
            try:
                out[k] = _coerce(k, v)
            except (TypeError, ValueError):
                continue
    return out


def _coerce(key: str, value: Any) -> Any:
    t = EDITABLE[key]
    if t is bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if t is int:
        return int(value)
    if t is float:
        return float(value)
    return str(value)


def save_overrides(patch: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    """Verilen alanlari kaydeder ve guncel tum override'lari dondurur.

    Bilinmeyen alanlar sessizce yok sayilir. Bos string gonderilen sir
    alanlari (or. llm_api_key) mevcut degeri KORUR (kullanici alani bos
    birakarak kaydettiginde anahtar silinmesin diye).
    """
    p = path or store_path()
    current = load_overrides(p)
    for k, v in patch.items():
        if k not in EDITABLE:
            continue
        if k in SECRET_FIELDS and (v is None or str(v).strip() == ""):
            continue
        if v is None:
            current.pop(k, None)
            continue
        current[k] = _coerce(k, v)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    return current


def clear_overrides(path: Path | None = None) -> None:
    p = path or store_path()
    if p.exists():
        p.unlink()


def mask_secret(value: str) -> str:
    if not value:
        return ""
    tail = value[-4:] if len(value) > 8 else ""
    return f"••••••{tail}"


def public_settings(overrides: dict[str, Any]) -> dict[str, Any]:
    """API'ye donulecek guvenli gorunum (sir maskeli)."""
    out: dict[str, Any] = {}
    for k, v in overrides.items():
        out[k] = mask_secret(str(v)) if k in SECRET_FIELDS else v
    return out
