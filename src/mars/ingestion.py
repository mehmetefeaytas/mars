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


def _read_cache(path: Path, ttl_hours: int) -> pd.DataFrame | None:
    if not path.exists():
        return None
    age_hours = (time.time() - path.stat().st_mtime) / 3600.0
    if age_hours > ttl_hours:
        return None
    try:
        return pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return None


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
) -> LoadResult:
    """Bir ticker icin gunluk OHLCV yukler (onbellek -> ag -> hata)."""
    settings = settings or load_settings()
    if start is None:
        start = (datetime.now(timezone.utc) - timedelta(days=settings.lookback_days)).strftime(
            "%Y-%m-%d"
        )
    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    cache = _cache_path(settings, ticker)
    cached = _read_cache(cache, settings.cache_ttl_hours)
    if cached is not None and not cached.empty:
        return LoadResult(frame=cached, source="cache", ticker=ticker)

    if allow_network:
        try:
            df = _fetch_network(ticker, start, end)
            _write_cache(cache, df)
            return LoadResult(frame=df, source="network", ticker=ticker)
        except Exception as exc:  # noqa: BLE001
            # Eski onbellek TTL disinda olsa bile son care olarak kullanilir.
            stale = None
            if cache.exists():
                try:
                    stale = pd.read_parquet(cache)
                except Exception:  # noqa: BLE001
                    stale = None
            if stale is not None and not stale.empty:
                return LoadResult(frame=stale, source="cache", ticker=ticker)
            raise RuntimeError(
                f"Veri alinamadi ({ticker}) ve onbellek yok: {exc}"
            ) from exc

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
    return joined.dropna(subset=["close", "market_close"])
