# SPDX-License-Identifier: AGPL-3.0-only
"""Haber analiz orkestrasyonu: polarite + guvenilirlik -> NewsSignal."""

from __future__ import annotations

from ..config import Settings, load_settings
from .sentiment import analyze_sentiment
from .sources import NewsItem, NewsSignal
from .verishield_adapter import score_credibility


def analyze_news(item: NewsItem, settings: Settings | None = None) -> NewsSignal:
    """Bir haber kalemini polarite + guvenilirlik ile zenginlestirir."""
    settings = settings or load_settings()
    sentiment = analyze_sentiment(
        item.text, backend=settings.sentiment_backend, model_name=settings.sentiment_model
    )
    credibility = score_credibility(
        item.text, source=item.source, settings=settings
    )
    # Resmi kaynaklar (KAP) en azindan yapisal olarak daha guvenilirdir.
    if item.official:
        credibility.score = max(credibility.score, 0.85)
        credibility.factors.append("Resmi kaynak (KAP) bildirimi.")
    return NewsSignal(item=item, sentiment=sentiment, credibility=credibility)
