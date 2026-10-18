# SPDX-License-Identifier: AGPL-3.0-only
"""MARS HTTP API — React frontend'in servis katmani.

Calistirma:
    PYTHONPATH=src .venv/bin/uvicorn mars.api:app --port 8010

Uclar:
    GET  /api/health                     -> durum
    GET  /api/meta                       -> status/varsayilan parametreler/sepet
    GET  /api/analyze?ticker=THYAO.IS    -> tam analiz (JSON)
    POST /api/screener                   -> sepet taramasi
    GET  /api/news?limit=60              -> RSS haberleri (analizli, borsa-ilgili)
    POST /api/explain                    -> LLM / kural-tabanli gun aciklamasi
    GET/PUT /api/settings                -> calisma-zamani ayarlari
    GET  /api/settings/models            -> saglayicidan model listesi
    POST /api/settings/llm/test          -> LLM baglanti testi
    GET  /api/settings/terms             -> kullanim sartlari durumu (EVREN)
    POST /api/settings/terms/accept      -> sartlari onayla
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import __version__
from .config import FusionWeights, Settings, load_settings
from .explain import detailed_reasons, summarize_day, waterfall
from .fusion.weighted import fuse
from .news.analyzer import analyze_news
from .news.filters import filter_market_news, market_category
from .news.llm import (
    LLMError,
    LLMTermsRequired,
    accept_terms,
    explain_with_llm,
    fetch_models,
    llm_available,
    terms_status,
)
from .news.rss import fetch_all_feeds, news_for_ticker
from .news.sources import NewsSignal
from .news.tickers import match_ticker
from .news.verishield_adapter import verishield_available
from . import ml as ml_module
from .pipeline import analyze

app = FastAPI(title="MARS API", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8010"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sepet: 12 likit baslangic hissesi + kullanici izleme listesi (v3)
DEFAULT_BASKET = [
    # --- Buyuk olcekli / likit ---
    "THYAO.IS", "GARAN.IS", "AKBNK.IS", "ASELS.IS", "EREGL.IS", "SISE.IS",
    "KCHOL.IS", "BIMAS.IS", "TUPRS.IS", "PETKM.IS", "SASA.IS", "FROTO.IS",
    # --- Izleme listesi (v3) ---
    "ALKLC.IS", "ANELE.IS", "BIGEN.IS", "BTCIM.IS", "BURCE.IS", "BURVA.IS",
    "CELHA.IS", "DMRGD.IS", "DSTKF.IS", "EDATA.IS", "GMTAS.IS", "GRTHO.IS",
    "GUNDG.IS", "HEDEF.IS", "IEYHO.IS", "INFO.IS", "KERVN.IS", "KRDMD.IS",
    "KTLEV.IS", "KUYAS.IS", "MANAS.IS", "ODINE.IS", "OZATD.IS", "OZSUB.IS",
    "PASEU.IS", "PEKGY.IS", "PRZMA.IS", "SELEC.IS", "TEHOL.IS", "TRHOL.IS",
    "VSNMD.IS", "EYGYO.IS", "AHGAZ.IS", "TERA.IS", "BVSAN.IS", "A1CAP.IS",
    "SNICA.IS", "KAYSE.IS", "OBASE.IS", "SOKE.IS",
]


# --------------------------------------------------------------------------
# Istek modelleri
# --------------------------------------------------------------------------
class AnalyzeParams(BaseModel):
    lookback: int = Field(400, ge=120, le=1000)
    rolling: int = Field(60, ge=20, le=120)
    est_window: int = Field(120, ge=60, le=250)
    iforest_trees: int = Field(200, ge=50, le=500)
    contamination: float = Field(0.02, ge=0.005, le=0.10)
    news_penalty: float = Field(0.35, ge=0.0, le=0.6)
    tahta_modulation: float = Field(1.0, ge=0.0, le=1.0)
    weights: dict[str, float] | None = None
    rss: bool = True
    llm: bool | None = None


class ScreenerRequest(BaseModel):
    tickers: list[str] = DEFAULT_BASKET
    params: AnalyzeParams = AnalyzeParams()


class ExplainRequest(BaseModel):
    ticker: str
    date: str | None = None
    params: AnalyzeParams = AnalyzeParams()


class SettingsPatch(BaseModel):
    llm_enabled: bool | None = None
    llm_provider: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None
    llm_timeout: int | None = None
    llm_thinking: bool | None = None
    llm_max_tokens: int | None = None
    llm_temperature: float | None = None
    rss_enabled: bool | None = None
    rss_feeds: str | None = None
    rss_max_items: int | None = None
    sentiment_backend: str | None = None
    verishield_enabled: bool | None = None
    lookback_days: int | None = None
    rolling_window: int | None = None
    estimation_window: int | None = None
    iforest_trees: int | None = None
    iforest_contamination: float | None = None
    news_penalty_strength: float | None = None
    tahta_modulation: float | None = None


class TermsAccept(BaseModel):
    version: int = 1


# --------------------------------------------------------------------------
# Onbellek (basit TTL)
# --------------------------------------------------------------------------
class TTLCache:
    def __init__(self, ttl_seconds: float = 900.0, max_items: int = 128):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._store: dict[Any, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: Any) -> Any | None:
        with self._lock:
            hit = self._store.get(key)
            if hit is None:
                return None
            ts, value = hit
            if time.time() - ts > self.ttl:
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: Any, value: Any) -> None:
        with self._lock:
            if len(self._store) >= self.max_items:
                oldest = min(self._store, key=lambda k: self._store[k][0])
                self._store.pop(oldest, None)
            self._store[key] = (time.time(), value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


_cache = TTLCache()
_news_cache = TTLCache(ttl_seconds=600.0)


def _clear_caches() -> None:
    _cache.clear()
    _news_cache.clear()


def _settings_from(params: AnalyzeParams, base: Settings | None = None) -> Settings:
    settings = base or load_settings()
    settings.lookback_days = params.lookback
    settings.rolling_window = params.rolling
    settings.estimation_window = params.est_window
    settings.iforest_trees = params.iforest_trees
    settings.iforest_contamination = params.contamination
    settings.news_penalty_strength = params.news_penalty
    settings.tahta_modulation = params.tahta_modulation
    if params.weights:
        known = set(FusionWeights.__dataclass_fields__)
        kwargs = {k: float(v) for k, v in params.weights.items() if k in known}
        if kwargs:
            settings.weights = FusionWeights(**{**settings.weights.as_dict(), **kwargs})
    if params.llm is not None:
        settings.llm_enabled = params.llm
    return settings


def _nan_to_none(values: list) -> list:
    return [None if (isinstance(v, float) and not np.isfinite(v)) else v for v in values]


def _series(frame: pd.DataFrame, col: str) -> list:
    if col not in frame.columns:
        return [None] * len(frame)
    return _nan_to_none(frame[col].tolist())


# --------------------------------------------------------------------------
# Haber yardimcilari
# --------------------------------------------------------------------------
def _news_payload(items: list, settings: Settings) -> list[dict]:
    out = []
    for i in items:
        sig = analyze_news(i, settings)
        out.append({
            "headline": i.headline,
            "body": i.body[:400],
            "source": i.source,
            "url": i.url,
            "timestamp": None if i.timestamp is None else i.timestamp.isoformat(),
            "polarity": round(sig.sentiment.polarity, 3),
            "label": sig.sentiment.label,
            "confidence": round(sig.sentiment.confidence, 3),
            "sentiment_backend": sig.sentiment.backend,
            "credibility": round(sig.credibility.score, 3),
            "credibility_backend": sig.credibility.backend,
            "category": market_category(i.text) or "diger",
            "ticker": match_ticker(i.text),
        })
    return out


def _get_news(limit: int) -> list[dict]:
    """Borsa-ilgili RSS haberleri (analizli, onbellekli)."""
    key = ("news", limit)
    cached = _news_cache.get(key)
    if cached is not None:
        return cached
    settings = load_settings()
    items, _errors = fetch_all_feeds(settings)
    items = filter_market_news(items)[:limit]
    payload = _news_payload(items, settings)
    _news_cache.set(key, payload)
    return payload


def _news_signals_for(ticker: str, settings: Settings) -> tuple[list[NewsSignal], list[dict]]:
    """Bir hisseyle eslesen haberler -> (NewsSignal listesi, JSON temsili)."""
    items, _errors = fetch_all_feeds(settings)
    matched = news_for_ticker(ticker, items)[:8]
    signals: list[NewsSignal] = []
    payload: list[dict] = []
    for item in matched:
        sig = analyze_news(item, settings)
        signals.append(sig)
        payload.append({
            "headline": item.headline,
            "source": item.source,
            "url": item.url,
            "timestamp": None if item.timestamp is None else item.timestamp.isoformat(),
            "polarity": round(sig.sentiment.polarity, 3),
            "label": sig.sentiment.label,
            "credibility": round(sig.credibility.score, 3),
            "general": False,
        })
    return signals, payload


def _group_news(signals: list[NewsSignal]) -> dict[pd.Timestamp, list[NewsSignal]] | None:
    out: dict[pd.Timestamp, list[NewsSignal]] = {}
    for sig in signals:
        if sig.item.timestamp is not None:
            out.setdefault(pd.Timestamp(sig.item.timestamp).normalize(), []).append(sig)
    return out or None


# --------------------------------------------------------------------------
# Analiz cekirdegi
# --------------------------------------------------------------------------
def _run_analysis(ticker: str, params: AnalyzeParams, settings: Settings) -> dict:
    cache_key = ("analyze", ticker, params.model_dump_json())
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    result = analyze(ticker, settings=settings, news_by_date=None)  # agir kisim
    sf = result.signal_frame

    news_signals: list[NewsSignal] = []
    news_out: list[dict] = []
    if params.rss and settings.rss_enabled:
        news_signals, news_out = _news_signals_for(ticker, settings)

    news_is_general = False
    if not news_out:
        # Hisseyle eslesen haber yoksa: hisse sayfasinda kaybolmasin diye
        # en guncel piyasa haberleri gosterilir (isaretlenir).
        general = _get_news(12)[:8]
        news_out = [
            {
                "headline": g["headline"],
                "source": g["source"],
                "url": g["url"],
                "timestamp": g["timestamp"],
                "polarity": g["polarity"],
                "label": g["label"],
                "credibility": g["credibility"],
                "general": True,
            }
            for g in general
        ]
        news_is_general = bool(news_out)

    fusion = fuse(
        sf,
        news_by_date=_group_news(news_signals),
        abnormal_returns=sf["abnormal_ret"],
        settings=settings,
    )
    frame = sf.join(fusion.frame, how="left")

    last_idx = len(frame) - 1
    srow, frow = frame.iloc[last_idx], frame.iloc[last_idx]
    payload = {
        "ticker": ticker,
        "data_source": result.data_source,
        "ml_available": bool(sf["sig_ml"].notna().any()) if "sig_ml" in sf.columns else False,
        "dates": [d.strftime("%Y-%m-%d") for d in frame.index],
        "open": _series(frame, "open"),
        "high": _series(frame, "high"),
        "low": _series(frame, "low"),
        "close": _series(frame, "close"),
        "volume": _series(frame, "volume"),
        "suspicion": _series(frame, "suspicion"),
        "suspicion_base": _series(frame, "suspicion_base"),
        "data_confidence": _series(frame, "data_confidence"),
        "news_factor": _series(frame, "news_factor"),
        "log_ret": _series(frame, "log_ret"),
        "abnormal_ret": _series(frame, "abnormal_ret"),
        "car5": _series(frame, "car5"),
        "clv": _series(frame, "clv"),
        "signals": {c: _series(frame, c) for c in frame.columns if c.startswith("sig_")},
        "contributions": {c: _series(frame, c) for c in frame.columns if c.startswith("contrib_")},
        "tahta_prior": round(float(sf["tahta_prior"].iloc[-1]), 3),
        "liquidity_tier": str(sf["liquidity_tier"].iloc[-1]),
        "adv_dollar_median": float(sf["adv_dollar_median"].iloc[-1]),
        "news": news_out,
        "news_is_general": news_is_general,
        "summary": summarize_day(frow, date=frame.index[-1], signal_row=srow),
        "waterfall": waterfall(frow, top_k=9),
        "reasons": detailed_reasons(srow, frow),
    }
    _cache.set(cache_key, payload)
    return payload


# --------------------------------------------------------------------------
# Uclar — durum / sepet
# --------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/api/meta")
def meta() -> dict:
    settings = load_settings()
    model = ml_module.load_model(settings)
    return {
        "version": __version__,
        "verishield": verishield_available(),
        "ml": {
            "available": model is not None,
            "auc": round(model.auc, 4) if model else None,
            "coefficients": model.artifact["meta"]["coefficients"] if model else {},
        },
        "llm": {
            "enabled": settings.llm_enabled,
            "available": llm_available(settings) if settings.llm_enabled else False,
            "model": settings.llm_model,
            "provider": settings.llm_provider,
        },
        "rss": {"enabled": settings.rss_enabled, "feeds": settings.rss_feeds.split(",")},
        "basket": DEFAULT_BASKET,
        "defaults": {
            "params": AnalyzeParams().model_dump(),
            "weights": settings.weights.as_dict(),
        },
        "screener_basket": DEFAULT_BASKET,
    }


@app.get("/api/analyze")
def api_analyze(
    ticker: str = Query(..., description="BIST ticker, or. THYAO.IS"),
    lookback: int = 400,
    rolling: int = 60,
    est_window: int = 120,
    iforest_trees: int = 200,
    contamination: float = 0.02,
    news_penalty: float = 0.35,
    tahta_modulation: float = 1.0,
    rss: bool = True,
) -> dict:
    params = AnalyzeParams(
        lookback=lookback, rolling=rolling, est_window=est_window,
        iforest_trees=iforest_trees, contamination=contamination,
        news_penalty=news_penalty, tahta_modulation=tahta_modulation, rss=rss,
    )
    settings = _settings_from(params)
    try:
        return _run_analysis(ticker.upper(), params, settings)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Analiz basarisiz: {exc}") from exc


@app.post("/api/screener")
def api_screener(req: ScreenerRequest) -> dict:
    settings = _settings_from(req.params)
    rows = []
    errors = []
    for t in req.tickers:
        t = t.strip().upper()
        if not t:
            continue
        try:
            payload = _run_analysis(t, req.params, settings)
            last = len(payload["dates"]) - 1
            rows.append({
                "ticker": t,
                "suspicion": payload["suspicion"][last],
                "data_confidence": payload["data_confidence"][last],
                "log_ret": payload["log_ret"][last],
                "abnormal_ret": payload["abnormal_ret"][last],
                "top_signal": payload["waterfall"][0]["signal"] if payload["waterfall"] else "—",
                "top_points": payload["waterfall"][0]["points"] if payload["waterfall"] else 0,
                "liquidity_tier": payload["liquidity_tier"],
                "tahta_prior": payload["tahta_prior"],
                "news_count": sum(1 for n in payload["news"] if not n.get("general")),
                "date": payload["dates"][last],
            })
        except Exception as exc:  # noqa: BLE001
            errors.append({"ticker": t, "error": str(exc)[:120]})
    rows.sort(key=lambda r: (r["suspicion"] is None, -(r["suspicion"] or 0)))
    return {"rows": rows, "errors": errors}


@app.get("/api/news")
def api_news(limit: int = 60) -> dict:
    items = _get_news(limit)
    matched: dict[str, int] = {}
    for it in items:
        t = it.get("ticker")
        if t:
            matched[t] = matched.get(t, 0) + 1
    return {"items": items, "matched": matched}


# --------------------------------------------------------------------------
# Uclar — aciklama
# --------------------------------------------------------------------------
@app.post("/api/explain")
def api_explain(req: ExplainRequest) -> dict:
    settings = _settings_from(req.params)
    cache_key = ("explain", req.ticker.upper(), req.date, req.params.model_dump_json(),
                 settings.llm_model, settings.llm_provider)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    ticker = req.ticker.strip().upper()
    try:
        result = analyze(ticker, settings=settings, news_by_date=None)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Analiz basarisiz: {exc}") from exc

    news_signals, _payload = (
        _news_signals_for(ticker, settings)
        if req.params.rss and settings.rss_enabled else ([], [])
    )
    fusion = fuse(
        result.signal_frame,
        news_by_date=_group_news(news_signals),
        abnormal_returns=result.signal_frame["abnormal_ret"],
        settings=settings,
    )
    frame = result.signal_frame.join(fusion.frame, how="left")

    dates = [d.strftime("%Y-%m-%d") for d in frame.index]
    date = req.date or dates[-1]
    ts = pd.Timestamp(date)
    if ts not in frame.index:
        raise HTTPException(status_code=400, detail=f"Tarih bulunamadi: {date}")

    srow, frow = frame.loc[ts], frame.loc[ts]
    reasons = detailed_reasons(srow, frow)

    llm_text = None
    llm_error = None
    if settings.llm_enabled:
        lr = srow.get("log_ret", float("nan"))
        ar = srow.get("abnormal_ret", float("nan"))
        llm_payload = {
            "ticker": ticker,
            "date": date,
            "suspicion": round(float(frow["suspicion"]), 1),
            "data_confidence": round(float(frow["data_confidence"]), 1),
            "log_ret": f"{lr*100:+.2f}%" if np.isfinite(lr) else "—",
            "abnormal_ret": f"{ar*100:+.2f}%" if np.isfinite(ar) else "—",
            "liquidity_tier": str(srow.get("liquidity_tier", "—")),
            "reasons": reasons,
            "news": [
                {"source": n.item.source, "headline": n.item.headline,
                 "polarity": n.sentiment.polarity, "credibility": n.credibility.score}
                for n in news_signals
            ],
        }
        try:
            llm_text, llm_error = explain_with_llm(llm_payload, settings)
        except LLMTermsRequired as exc:
            raise HTTPException(
                status_code=409,
                detail=(f"LLM kullanim sartlari onaylanmali (surum "
                        f"{exc.version if exc.version else '?'}). Ayarlar sayfasindan onaylayin."),
            ) from exc
        except LLMError as exc:
            llm_error = str(exc)

    out = {
        "ticker": ticker,
        "date": date,
        "suspicion": round(float(frow["suspicion"]), 1),
        "data_confidence": round(float(frow["data_confidence"]), 1),
        "summary": summarize_day(frow, date=ts, signal_row=srow),
        "reasons": reasons,
        "llm": llm_text,
        "llm_backend": "llm" if llm_text else "kural-tabanli",
        "llm_error": llm_error,
    }
    _cache.set(cache_key, out)
    return out


# --------------------------------------------------------------------------
# Uclar — ayarlar
# --------------------------------------------------------------------------
_PROVIDER_PRESETS = [
    {"id": "ollama", "label": "Yerel (Ollama)", "default_url": "http://localhost:11434/v1",
     "needs_key": False,
     "hint": "Bilgisayarinizda calisan model. Anahtar gerekmez."},
    {"id": "evren", "label": "EVREN LLM Gateway", "default_url": "https://evren-llmapi.ssyz.org.tr/v1",
     "needs_key": True,
     "hint": "X-API-Key ile calisir; ilk kullanimda kullanim sartlari onayi gerekir."},
    {"id": "openai", "label": "OpenAI", "default_url": "https://api.openai.com/v1",
     "needs_key": True, "hint": "Bearer API anahtari ile calisir."},
    {"id": "custom", "label": "Diger (OpenAI uyumlu)", "default_url": "",
     "needs_key": True, "hint": "OpenAI uyumlu herhangi bir sunucu."},
]


def _effective_settings(settings: Settings) -> dict:
    from .settings_store import mask_secret  # noqa: PLC0415

    return {
        "llm_enabled": settings.llm_enabled,
        "llm_provider": settings.llm_provider,
        "llm_base_url": settings.llm_base_url,
        "llm_model": settings.llm_model,
        "llm_api_key": mask_secret(settings.llm_api_key),
        "llm_api_key_set": bool(settings.llm_api_key),
        "llm_timeout": settings.llm_timeout,
        "llm_thinking": settings.llm_thinking,
        "llm_max_tokens": settings.llm_max_tokens,
        "llm_temperature": settings.llm_temperature,
        "rss_enabled": settings.rss_enabled,
        "rss_feeds": settings.rss_feeds,
        "rss_max_items": settings.rss_max_items,
        "sentiment_backend": settings.sentiment_backend,
        "verishield_enabled": settings.verishield_enabled,
        "lookback_days": settings.lookback_days,
        "rolling_window": settings.rolling_window,
        "estimation_window": settings.estimation_window,
        "iforest_trees": settings.iforest_trees,
        "iforest_contamination": settings.iforest_contamination,
        "news_penalty_strength": settings.news_penalty_strength,
        "tahta_modulation": settings.tahta_modulation,
    }


@app.get("/api/settings")
def get_settings() -> dict:
    from .settings_store import public_settings, store_path  # noqa: PLC0415

    settings = load_settings()
    terms: dict = {"supported": False, "accepted": True}
    if settings.llm_enabled and settings.llm_provider.lower() == "evren" and settings.llm_api_key:
        try:
            terms = terms_status(settings)
        except Exception as exc:  # noqa: BLE001
            terms = {"supported": True, "accepted": None, "error": str(exc)[:160]}
    return {
        "current": _effective_settings(settings),
        "overrides": public_settings(_overrides_view()),
        "providers": _PROVIDER_PRESETS,
        "store_path": str(store_path()),
        "evren_terms": terms,
        "llm_available": llm_available(settings) if settings.llm_enabled else False,
    }


def _overrides_view() -> dict:
    from .settings_store import load_overrides, store_path  # noqa: PLC0415

    return load_overrides(store_path())


@app.put("/api/settings")
def put_settings(patch: SettingsPatch) -> dict:
    from .settings_store import save_overrides, store_path  # noqa: PLC0415

    data = {k: v for k, v in patch.model_dump().items() if v is not None}
    if "llm_provider" in data:
        prov = str(data["llm_provider"]).lower()
        data["llm_provider"] = prov
        # Bos birakilirsa saglayicinin varsayilan URL'ini uygula
        if not data.get("llm_base_url"):
            for p in _PROVIDER_PRESETS:
                if p["id"] == prov and p["default_url"]:
                    data["llm_base_url"] = p["default_url"]
                    break
    save_overrides(data, store_path())
    _clear_caches()
    settings = load_settings()
    return {"current": _effective_settings(settings), "llm_available":
            llm_available(settings) if settings.llm_enabled else False}


@app.get("/api/settings/models")
def get_models() -> dict:
    settings = load_settings()
    try:
        return {"models": fetch_models(settings), "provider": settings.llm_provider}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Model listesi alinamadi: {exc}") from exc


@app.post("/api/settings/llm/test")
def llm_test() -> dict:
    from .news.llm import LLMError as _LLMError  # noqa: PLC0415
    from .news.llm import chat  # noqa: PLC0415

    settings = load_settings()
    if not settings.llm_enabled:
        return {"ok": False, "message": "LLM kapali. Once etkinlestirin."}
    try:
        text = chat(
            [{"role": "user", "content": "Tek cumleyle 'MARS testi basarili' yaz."}],
            settings, temperature=0.0, max_tokens=64,
        )
        cleaned = (text or "").strip()[:200]
        return {"ok": bool(cleaned), "message": cleaned or "Bos yanit.",
                "provider": settings.llm_provider, "model": settings.llm_model}
    except LLMTermsRequired as exc:
        return {"ok": False, "terms_required": True, "version": exc.version,
                "message": str(exc)}
    except _LLMError as exc:
        return {"ok": False, "message": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": str(exc)}


@app.get("/api/settings/terms")
def get_terms() -> dict:
    settings = load_settings()
    try:
        return terms_status(settings)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Sartlar sorgulanamadi: {exc}") from exc


@app.post("/api/settings/terms/accept")
def post_terms(req: TermsAccept) -> dict:
    settings = load_settings()
    try:
        return accept_terms(settings, req.version)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Sartlar onaylanamadi: {exc}") from exc


# Uretim: frontend build varsa SPA olarak servis et
try:
    from pathlib import Path

    from fastapi.responses import FileResponse

    _dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if _dist.exists():
        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str) -> Any:
            """SPA fallback: /api olmayan yollar index.html'e dus(er)."""
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Bulunamadi")
            candidate = (_dist / full_path).resolve()
            if full_path and candidate.exists() and candidate.is_file() and str(candidate).startswith(str(_dist)):
                return FileResponse(candidate)
            return FileResponse(_dist / "index.html")
except Exception:  # noqa: BLE001
    pass
