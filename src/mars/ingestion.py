# SPDX-License-Identifier: AGPL-3.0-only
"""Veri alimi — BIST gunluk OHLCV ve BIST100 endeksi.

Tasarim ilkeleri:
  * Ag erisimi (yfinance) birincil; ancak HER ZAMAN yerel Parquet onbellegi ve
    ornek arsiv fallback'i vardir. Boylece kesinti/lisans riski izole edilir.
  * Takvim hizalama: hisse ve endeks getirileri islem gunu bazinda birlestirilir.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .config import Settings, load_settings

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass
class LoadResult:
    frame: pd.DataFrame
    source: str  # "network" | "cache" | "sample" | "synthetic"
    ticker: str


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance ciktisini standart OHLCV semasina indirger."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    keep = [c for c in OHLCV_COLUMNS if c in df.columns]
    out = df[keep].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.index.name = "date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(how="all")


def _cache_path(settings: Settings, ticker: str) -> Path:
    safe = ticker.replace(".", "_").replace("/", "_")
    return settings.raw_dir / f"{safe}.parquet"


def _read_cache(
    path: Path, ttl_hours: int, start: str | None = None, end: str | None = None
) -> pd.DataFrame | None:
    """Onbellek gecerliyse okur; istenen tarih araligini KAPAMIYORSA reddeder.

    Sorun: ayni ticker icin farkli tarih araliklarinda yapilan cagirilar ayni
    parquet uzerine yazilir. Kapsam kontrolu olmazsa eski aralik doner ve hisse/
    endeks hizalama bosalir ("Hizalanmis veri bos" hatasi).
    """
    if not path.exists():
        return None
    age_hours = (time.time() - path.stat().st_mtime) / 3600.0
    if age_hours > ttl_hours:
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return None
    if df is None or df.empty:
        return None
    if start is not None:
        cs = df.index.min()
        want_start = pd.Timestamp(start)
        if cs > want_start + pd.Timedelta(days=7):
            return None  # onbellek istenen baslangici kapsamiyor
    if end is not None:
        ce = df.index.max()
        want_end = pd.Timestamp(end)
        if ce < want_end - pd.Timedelta(days=7):
            return None  # onbellek istenen bitisi kapsamiyor
    return df


def _write_cache(path: Path, df: pd.DataFrame, existing: pd.DataFrame | None = None) -> None:
    """Onbellege yazar; varsa eski veriyle birlestirir (union)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if existing is not None and not existing.empty:
        df = pd.concat([existing, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(path)


def _fetch_network(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf  # lokal import: paket yoksa fallback devreye girer

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance bos sonuc dondu: {ticker}")
    return _normalize_frame(raw)


def load_ohlcv(
    ticker: str,
    settings: Settings | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
    allow_network: bool = True,
    force_refresh: bool = False,
) -> LoadResult:
    """Bir ticker icin gunluk OHLCV yukler (onbellek -> ag -> bayat onbellek)."""
    settings = settings or load_settings()
    if start is None:
        start = (datetime.now(timezone.utc) - timedelta(days=settings.lookback_days)).strftime(
            "%Y-%m-%d"
        )
    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    cache = _cache_path(settings, ticker)
    existing: pd.DataFrame | None = None
    if cache.exists():
        try:
            existing = pd.read_parquet(cache)
        except Exception:  # noqa: BLE001
            existing = None

    if not force_refresh:
        cached = _read_cache(cache, settings.cache_ttl_hours, start, end)
        if cached is not None and not cached.empty:
            return LoadResult(frame=cached, source="cache", ticker=ticker)

    if allow_network:
        try:
            df = _fetch_network(ticker, start, end)
            _write_cache(cache, df, existing)
            return LoadResult(frame=df, source="network", ticker=ticker)
        except Exception as exc:  # noqa: BLE001
            # Ag hatasinda: kapsam disi bile olsa bayat onbellek son caredir.
            if existing is not None and not existing.empty:
                return LoadResult(frame=existing, source="cache-stale", ticker=ticker)
            raise RuntimeError(
                f"Veri alinamadi ({ticker}) ve kullanilabilir onbellek yok: {exc}"
            ) from exc

    if existing is not None and not existing.empty:
        return LoadResult(frame=existing, source="cache-stale", ticker=ticker)
    raise RuntimeError(f"Ag kapali ve onbellek yok: {ticker}")


def load_bist100(settings: Settings | None = None, **kwargs) -> LoadResult:
    settings = settings or load_settings()
    return load_ohlcv(settings.benchmark_ticker, settings, **kwargs)


def align_stock_market(stock: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Hisse ve endeks getirilerini islem gunu bazinda hizalar.

    Donen cerceve: hisse OHLCV + market_close sutunlari, kesisim gunlerinde.
    """
    stock = stock.copy()
    market = market.copy()
    stock.index.name = "date"
    market.index.name = "date"
    joined = stock.join(market[["close"]].rename(columns={"close": "market_close"}), how="inner")
    joined = joined.dropna(subset=["close", "market_close"])
    if joined.empty:
        raise RuntimeError(
            "Hisse ve endeks verilerinin tarih araliklari kesismiyor "
            f"(hisse: {stock.index.min().date()}–{stock.index.max().date()}, "
            f"endeks: {market.index.min().date()}–{market.index.max().date()}). "
            "Onbellek temizlenip yeniden cekilecek."
        )
    return joined
