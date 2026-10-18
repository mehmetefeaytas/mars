# SPDX-License-Identifier: AGPL-3.0-only
"""Haber modulu testleri (lexicon duygu, guvenilirlik fallback, VeriShield)."""

from mars.news.sentiment import analyze_sentiment, lexicon_sentiment
from mars.news.verishield_adapter import heuristic_credibility, score_credibility
from mars.news.analyzer import analyze_news
from mars.news.sources import NewsItem


def test_lexicon_positive():
    s = lexicon_sentiment("Şirket rekor kâr açıkladı, hisse yükseldi.")
    assert s.polarity > 0
    assert s.label == "pozitif"


def test_lexicon_negative():
    s = lexicon_sentiment("Şirket hakkında soruşturma açıldı, hisse düştü ve zarar açıklandı.")
    assert s.polarity < 0
    assert s.label == "negatif"


def test_lexicon_neutral():
    s = lexicon_sentiment("Yönetim kurulu olağan toplantısını yaptı.")
    assert s.label == "notr"


def test_negation_flips_polarity():
    pos = lexicon_sentiment("kâr var")
    neg = lexicon_sentiment("kâr yok")
    assert pos.polarity > neg.polarity


def test_analyze_sentiment_auto_falls_back():
    # torch yoksa lexicon'a dusmeli; backend adi bilgilendirici.
    s = analyze_sentiment("Şirket güçlü büyüme açıkladı.")
    assert s.backend in {"bert", "lexicon", "lexicon(bert-unavailable)"}
    assert s.polarity > 0


def test_heuristic_credibility_source_and_sensational():
    trusted = heuristic_credibility("Net kâr arttı, temettü açıklandı.", source="kap")
    shady = heuristic_credibility("Flaş iddia: hisse uçacak, kaçırmayın!", source="twitter")
    assert trusted.score > shady.score
    assert "sansasyonel_dil" in shady.flags


def test_score_credibility_returns_valid_range():
    c = score_credibility("Şirket bilanço açıkladı.", source="bloomberght")
    assert 0.0 <= c.score <= 1.0


def test_news_value():
    sig = analyze_news(NewsItem(headline="Şirket rekor kâr açıkladı", source="kap", official=True))
    assert sig.news_value > 0
