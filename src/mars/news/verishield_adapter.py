# SPDX-License-Identifier: AGPL-3.0-only
"""VeriShield entegrasyonu (gomulu kutuphane).

VeriShield, domain-bagimsiz dezenformasyon dogrulama motorudur (AGPL-3.0) ve
0-100 `guvenilirlik_skoru` uretir. MARS bu skoru haber agirligi icin kullanir.

Iki mod:
  * embedded (varsayilan): vendor/verishield-core/backend sys.path'e eklenir,
    izole bir SQLite ile `pipeline.process` cagrilir.
  * http: calisan VeriShield sunucusuna POST /api/ingest.

Her iki mod da basarisiz olursa yerel sezgisel bir skorlayici (heuristic) devreye
girer; boylece MARS, VeriShield kurulu olmadan da calisir.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ..config import Settings, load_settings
from .sources import CredibilityResult

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_BACKEND = _REPO_ROOT / "vendor" / "verishield-core" / "backend"

_INITIALIZED = False
_SESSION_FACTORY = None
_PIPELINE = None
_REPORTIN = None
_INIT_ERROR: str | None = None


def verishield_available() -> bool:
    return _VENDOR_BACKEND.exists()


def _init_embedded() -> bool:
    """VeriShield'i gomulu olarak hazirlar. Basariliysa True."""
    global _INITIALIZED, _SESSION_FACTORY, _PIPELINE, _REPORTIN, _INIT_ERROR
    if _INITIALIZED:
        return _INIT_ERROR is None
    _INITIALIZED = True

    if not verishield_available():
        _INIT_ERROR = "vendor/verishield-core bulunamadi"
        return False

    # KRITIK: verishield_core import edilmeden ONCE ortam ayarlanmali.
    db_path = Path(tempfile.gettempdir()) / "mars_verishield.db"
    os.environ.setdefault("DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    os.environ.setdefault("BGE_ENABLED", "false")
    os.environ.setdefault("LLM_ENABLED", "false")
    os.environ.setdefault("AUTH_ENABLED", "false")

    backend = str(_VENDOR_BACKEND)
    if backend not in sys.path:
        sys.path.insert(0, backend)

    try:
        from verishield_core import pipeline  # noqa: PLC0415
        from verishield_core.db import Base, SessionLocal, engine  # noqa: PLC0415
        from verishield_core.schemas import ReportIn  # noqa: PLC0415

        Base.metadata.create_all(engine)
        _SESSION_FACTORY = SessionLocal
        _PIPELINE = pipeline
        _REPORTIN = ReportIn
        return True
    except Exception as exc:  # noqa: BLE001
        _INIT_ERROR = f"{type(exc).__name__}: {exc}"
        return False


def _score_embedded(text: str, source: str, domain: str = "genel-haber") -> CredibilityResult | None:
    if not _init_embedded():
        return None
    report_id = f"mars-{abs(hash(text)) % 10_000_000}"
    try:
        report = _REPORTIN(
            id=report_id,
            domain=domain,
            kaynak="haber",
            kaynak_id=source or "unknown",
            metin=text,
            zaman=datetime.now(timezone.utc),
        )
        with _SESSION_FACTORY() as session:  # type: ignore[misc]
            out = _PIPELINE.process(session, report)
        score = float(out.guvenilirlik_skoru) / 100.0
        return CredibilityResult(
            score=round(score, 4),
            factors=list(out.gerekce_ozeti or [])[:8],
            flags=list(out.bayraklar or []),
            backend="verishield",
        )
    except Exception:  # noqa: BLE001
        return None


def _score_http(text: str, source: str, settings: Settings) -> CredibilityResult | None:
    import requests

    url = f"{settings.verishield_base_url.rstrip('/')}/api/ingest"
    payload = {
        "id": f"mars-{abs(hash(text)) % 10_000_000}",
        "domain": "genel-haber",
        "kaynak": "haber",
        "kaynak_id": source or "unknown",
        "metin": text,
        "zaman": datetime.now(timezone.utc).isoformat(),
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
        data = r.json().get("score", {})
        return CredibilityResult(
            score=round(float(data.get("guvenilirlik_skoru", 50)) / 100.0, 4),
            factors=list(data.get("gerekce_ozeti", []) or [])[:8],
            flags=list(data.get("bayraklar", []) or []),
            backend="http",
        )
    except Exception:  # noqa: BLE001
        return None


# --- Yerel sezgisel fallback -------------------------------------------------

_TRUSTED_SOURCES = {"kap", "reuters", "bloomberght", "bloomberg", "anadolu", "aa", "matriks", "foreks"}
_SENSATIONAL = {"iddia", "soylenti", "söylenti", "dedikodu", "flas", "flaş", "kesin", "garanti", "uçacak", "ucacak", "kaçırmayın", "kacirmayin"}


def heuristic_credibility(text: str, source: str = "") -> CredibilityResult:
    """Kaynak guvenilirligi + sansasyonel dil sezgisi (VeriShield yoksa)."""
    factors: list[str] = []
    flags: list[str] = []
    score = 0.5

    src = (source or "").lower()
    if any(t in src for t in _TRUSTED_SOURCES):
        score += 0.25
        factors.append("Kaynak bilinen/guvenilir kurumsal kanal.")
    if src == "" or src in {"unknown", "anonymous"}:
        score -= 0.1
        factors.append("Kaynak belirtilmemis; guven azaldi.")

    lowered = text.lower()
    hits = [w for w in _SENSATIONAL if w in lowered]
    if hits:
        score -= 0.08 * len(hits)
        flags.append("sansasyonel_dil")
        factors.append(f"Sansasyonel ifadeler: {', '.join(hits[:4])}.")

    # Sayisal/tarihli iddialar daha dogrulanabilirdir.
    import re

    if re.search(r"\d", text):
        score += 0.08
        factors.append("Sayisal/tarihli ifade iceriyor (dogrulanabilirlik artar).")

    score = max(0.05, min(0.95, score))
    return CredibilityResult(score=round(score, 4), factors=factors, flags=flags, backend="heuristic")


def score_credibility(
    text: str, *, source: str = "", settings: Settings | None = None
) -> CredibilityResult:
    """VeriShield (gomulu/http) ile guvenilirlik skoru; olmazsa sezgisel fallback."""
    settings = settings or load_settings()
    if settings.verishield_enabled:
        if settings.verishield_mode == "http":
            res = _score_http(text, source, settings)
        else:
            res = _score_embedded(text, source)
        if res is not None:
            return res
    return heuristic_credibility(text, source)
