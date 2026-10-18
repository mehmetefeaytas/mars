# SPDX-License-Identifier: AGPL-3.0-only
"""Haber suzgeci — "borsa disi" haberleri eler.

Kural: bir haber, (a) bir BIST sirketiyle/ticker'iyla eslesiyorsa VEYA
(b) piyasa/ekonomi anahtar kelimelerinden birini iceriyorsa "borsa ile ilgili"
sayilir. Boylece magazin, spor, saglik, genel siyaset gibi haberler elenirken
"Migros'un yeni magaza acmasi" gibi sirket haberleri korunur (hisse kodu
gecmese bile sirket adi eslesmesi yeterlidir).
"""

from __future__ import annotations

from .sources import NewsItem
from .tickers import match_ticker, normalize_tr

# Piyasa / ekonomi gostergeleri
MARKET_KEYWORDS: tuple[str, ...] = (
    "borsa", "bist", "borsa istanbul", "hisse", "endeks", "xu100", "borsada",
    "halka arz", "temettü", "temettu", "bedelli", "bedelsiz", "sermaye artır",
    "sermaye artir", "bilanço", "bilanco", "finansal tablo", "faiz", "enflasyon",
    "tcmb", "merkez bankası", "merkez bankasi", "faiz kararı", "dolar", "euro",
    "döviz", "doviz", "kur ", "altın", "altin", "gram altın", "petrol", "brent",
    "tahvil", "bono", "viop", "vadeli", "spot piyasa", "borsa endeksi",
    "yatırımcı", "yatirimci", "portföy", "portfoy", "fon ", "kredi", "mevduat",
    "işlem hacmi", "islem hacmi", "temettü", "piyasa", "piyasalar", "yükseliş",
    "yükselis", "düşüş", "dusus", "prim yaptı", "ralli", "çeyrek kâr",
    "çeyrek kar", "ciro", "hasılat", "net kâr", "net kar",
)

# Sirket faaliyeti / kurumsal haber gostergeleri
COMPANY_KEYWORDS: tuple[str, ...] = (
    "satın alma", "satin alma", "birleşme", "birlesme", "ortaklık", "ortaklik",
    "iştirak", "istirak", "ihale", "sözleşme", "sozlesme", "kap bildirimi",
    "kap'a", "yönetim kurulu", "yonetim kurulu", "genel kurul", "üretim",
    "uretim", "fabrika", "yatırım kararı", "yatirim karari", "kapasite artış",
    "kapasite artis", "yeni şube", "yeni sube", "mağaza aç", "magaza ac",
    "ihracat", "ithalat", "konkordato", "iflas", "bölünme", "bolunme",
    "kâr payı", "kar payı", "fiyat hedefi", "hedef fiyat", "analist",
    "değerleme", "degerleme", "not artır", "not artir", "brüt takas",
    "brut takas", "kotasyon", "işlem görmeye başla", "islem gormeye basla",
)


def market_category(text: str) -> str | None:
    """(hisse) | (piyasa) | (sirket) | None dondurur."""
    if not text:
        return None
    if match_ticker(text):
        return "hisse"
    norm = normalize_tr(text)
    if any(k in norm for k in MARKET_KEYWORDS):
        return "piyasa"
    if any(k in norm for k in COMPANY_KEYWORDS):
        return "sirket"
    return None


def is_market_related(text: str) -> bool:
    return market_category(text) is not None


def annotate(item: NewsItem) -> tuple[NewsItem, str, str | None]:
    """(item, kategori, ticker) dondurur."""
    cat = market_category(item.text) or "diger"
    return item, cat, match_ticker(item.text)


def filter_market_news(items: list[NewsItem]) -> list[NewsItem]:
    """Yalnizca borsa/ekonomi ile ilgili haberleri dondurur (sira korunur)."""
    return [i for i in items if is_market_related(i.text)]
