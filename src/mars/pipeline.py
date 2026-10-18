# SPDX-License-Identifier: AGPL-3.0-only
"""Uctan uca pipeline: veri -> ozellik -> sinyal -> füzyon -> aciklama."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import FusionWeights, Settings, load_settings
from .explain import summarize_day, waterfall
from .fusion.weighted import FusionResult, fuse
from .ingestion import align_stock_market, load_bist100, load_ohlcv
from .models.market_model import rolling_abnormal_returns
from .news.sources import NewsSignal
from .signals import SignalFrame, compute_signals


@dataclass
class AnalysisResult:
    ticker: str
    data_source: str
    signal_frame: pd.DataFrame
    fusion: FusionResult
    feature_columns: list[str] = field(default_factory=list)

    @property
    def frame(self) -> pd.DataFrame:
        """Sinyal + füzyon birlestirilmis cerceve (UI icin)."""
        return self.signal_frame.join(self.fusion.frame, how="left")

    def latest(self) -> pd.Series:
        return self.fusion.frame.iloc[-1]

    def summary(self, date: pd.Timestamp | None = None) -> str:
        idx = date if date is not None else self.fusion.frame.index[-1]
        return summarize_day(self.fusion.frame.loc[idx], date=idx)

    def waterfall(self, date: pd.Timestamp | None = None, top_k: int = 5) -> list[dict]:
        idx = date if date is not None else self.fusion.frame.index[-1]
        return waterfall(self.fusion.frame.loc[idx], top_k=top_k)


def analyze(
    ticker: str,
    *,
    settings: Settings | None = None,
    news_by_date: dict[pd.Timestamp, list[NewsSignal]] | None = None,
    benchmark: str | None = None,
    start: str | None = None,
    end: str | None = None,
    allow_network: bool = True,
    weights: FusionWeights | dict[str, float] | None = None,
) -> AnalysisResult:
    """Bir hisse icin tam MARS analizini kosturur."""
    settings = settings or load_settings()
    stock = load_ohlcv(ticker, settings, start=start, end=end, allow_network=allow_network)
    market = load_bist100(settings, start=start, end=end, allow_network=allow_network)
    aligned = align_stock_market(stock.frame, market.frame)
    if aligned.empty:
        raise RuntimeError(f"Hizalanmis veri bos: {ticker} / {settings.benchmark_ticker}")

    sf: SignalFrame = compute_signals(aligned, settings)
    ar = rolling_abnormal_returns(
        sf.frame["log_ret"], sf.frame["market_log_ret"], window=settings.estimation_window
    )
    fusion = fuse(
        sf.frame,
        news_by_date=news_by_date,
        abnormal_returns=ar,
        settings=settings,
        weights=weights,
    )
    return AnalysisResult(
        ticker=ticker,
        data_source=stock.source,
        signal_frame=sf.frame,
        fusion=fusion,
        feature_columns=sf.feature_columns,
    )
