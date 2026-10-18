# SPDX-License-Identifier: AGPL-3.0-only
"""Haber kaynak modeli.

MARS'ta haber sinyali iki eksende temsil edilir:
  * Guvenilirlik (credibility, VeriShield) — bilginin dogrulanabilirligi.
  * Polarite (sentiment) — haberin yonu ve siddeti.

KAP bildirimleri gibi resmi kaynaklar zaman damgasi + ikili bayrak olarak da
temsil edilebilir (rapordaki "haber var/yok" yaklasimi), ancak MARS zengin
metin sinyalini tercih eder ve geriye donuk uyumlulugu korur.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class NewsItem:
    headline: str
    body: str = ""
    source: str = ""
    ticker: str | None = None
    url: str | None = None
    timestamp: pd.Timestamp | None = None
    official: bool = False

    @property
    def text(self) -> str:
        return f"{self.headline}. {self.body}".strip()


@dataclass
class SentimentResult:
    polarity: float          # -1..1 (negatif..pozitif)
    label: str               # "pozitif" | "negatif" | "notr"
    confidence: float        # 0..1
    backend: str             # "bert" | "lexicon" | "llm"


@dataclass
class CredibilityResult:
    score: float             # 0..1 (yuksek = guvenilir)
    factors: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    backend: str = "heuristic"  # "verishield" | "heuristic" | "http"


@dataclass
class NewsSignal:
    item: NewsItem
    sentiment: SentimentResult
    credibility: CredibilityResult

    @property
    def polarity_signed(self) -> float:
        return self.sentiment.polarity

    @property
    def news_value(self) -> float:
        """Etkin haber degeri: polarite siddeti * guvenilirlik."""
        return abs(self.sentiment.polarity) * self.sentiment.confidence * self.credibility.score
