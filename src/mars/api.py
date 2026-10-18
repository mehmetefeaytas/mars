# SPDX-License-Identifier: AGPL-3.0-only
"""MARS HTTP API — React frontend'in servis katmani.

Calistirma:
    PYTHONPATH=src .venv/bin/uvicorn mars.api:app --port 8000 --reload

Uclar:
    GET  /api/health                     -> durum
    GET  /api/meta                       -> status/varsayilan parametreler/sepet
    GET  /api/analyze?ticker=THYAO.IS    -> tam analiz (JSON)
    POST /api/screener                   -> sepet taramasi
    GET  /api/news?limit=60              -> RSS haberleri (analizli)
    POST /api/explain                    -> LLM / kural-tabanli gun aciklamasi
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
from .news.llm import explain_with_llm, llm_available
from .news.rss import fetch_all_feeds, news_for_ticker
from .news.sources import NewsSignal
from .news.verishield_adapter import verishield_available
from . import ml as ml_module
from .pipeline import analyze

app = FastAPI(title="MARS API", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_BASKET = [
    "THYAO.IS", "GARAN.IS", "AKBNK.IS", "ASELS.IS", "EREGL.IS", "SISE.IS",
    "KCHOL.IS", "BIMAS.IS", "TUPRS.IS", "PETKM.IS", "SASA.IS", "FROTO.IS",
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


# --------------------------------------------------------------------------
# Onbellek (basit TTL)
# --------------------------------------------------------------------------
class TTLCache:
    def __init__(self, ttl_seconds: float = 900.0, max_items: int = 64):
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


_cache = TTLCache()
_news_cache = TTLCache(ttl_seconds=600.0)


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
def _get_news(limit: int) -> list[dict]:
    cached = _news_cache.get(("news", limit))
    if cached is not None:
        return cached
    settings = load_settings()
    items, _errors = fetch_all_feeds(settings)
    items = items[:limit]
    payload = []
    for i in items:
        sig = analyze_news(i, settings)
        payload.append({
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
            "ticker": None,
        })
    _news_cache.set(("news", limit), payload)
    return payload


def _news_signals_for(ticker: str, settings: Settings) -> tuple[list[NewsSignal], list[dict]]:
    """Bir hisseyle eslesen haberleri analiz et -> (NewsSignal listesi, JSON temsili)."""
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
    if params.rss and settings.rss_enabled:
        news_signals, _payload = _news_signals_for(ticker, settings)

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
        "news": [
            {
                "headline": n.item.headline,
                "source": n.item.source,
                "url": n.item.url,
                "timestamp": None if n.item.timestamp is None else n.item.timestamp.isoformat(),
                "polarity": round(n.sentiment.polarity, 3),
                "label": n.sentiment.label,
                "credibility": round(n.credibility.score, 3),
            }
            for n in news_signals
        ],
        "summary": summarize_day(frow, date=frame.index[-1], signal_row=srow),
        "waterfall": waterfall(frow, top_k=9),
        "reasons": detailed_reasons(srow, frow),
    }
    _cache.set(cache_key, payload)
    return payload


# --------------------------------------------------------------------------
# Uclar
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
                "news_count": len(payload["news"]),
                "date": payload["dates"][last],
            })
        except Exception as exc:  # noqa: BLE001
            errors.append({"ticker": t, "error": str(exc)[:120]})
    rows.sort(key=lambda r: (r["suspicion"] is None, -(r["suspicion"] or 0)))
    return {"rows": rows, "errors": errors}


@app.get("/api/news")
def api_news(limit: int = 60) -> dict:
    items = _get_news(limit)
    matched = auto_match_from_payload(items)
    return {"items": items, "matched": matched}


def auto_match_from_payload(items: list[dict]) -> dict[str, int]:
    from .news.tickers import match_ticker  # noqa: PLC0415

    out: dict[str, int] = {}
    for it in items:
        t = match_ticker(f"{it.get('headline', '')} {it.get('body', '')}")
        if t:
            out[t] = out.get(t, 0) + 1
    return out


@app.post("/api/explain")
def api_explain(req: ExplainRequest) -> dict:
    settings = _settings_from(req.params)
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
    if settings.llm_enabled and llm_available(settings):
        lr = srow.get("log_ret", float("nan"))
        ar = srow.get("abnormal_ret", float("nan"))
        llm_payload = {
            "ticker": ticker,
            "date": date,
            "suspicion": round(float(frow["suspicion"]), 1),
            "data_confidence": round(float(frow["data_confidence"]), 1),
            "log_ret": f"{lr*100:+.2f}%" if np.isfinite(lr) else "—",
            "abnormal_ret": f"{ar*100:+.2f}%" if np.isfinite(ar) else "—",
            "reasons": reasons,
            "news": [
                {"source": n.item.source, "headline": n.item.headline,
                 "polarity": n.sentiment.polarity, "credibility": n.credibility.score}
                for n in news_signals
            ],
        }
        llm_text = explain_with_llm(llm_payload, settings)

    return {
        "ticker": ticker,
        "date": date,
        "suspicion": round(float(frow["suspicion"]), 1),
        "data_confidence": round(float(frow["data_confidence"]), 1),
        "summary": summarize_day(frow, date=ts, signal_row=srow),
        "reasons": reasons,
        "llm": llm_text,
        "llm_backend": "ollama" if llm_text else "kural-tabanli",
    }


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
