# SPDX-License-Identifier: AGPL-3.0-only
"""Haber suzgeci ve ticker eslestirme testleri."""

from __future__ import annotations

from mars.news.filters import filter_market_news, is_market_related, market_category
from mars.news.sources import NewsItem
from mars.news.tickers import match_text_for_ticker, match_ticker


def _item(text: str) -> NewsItem:
    return NewsItem(headline=text, body="", source="test")


# --- Piyasa disi haberler elenmeli ---
def test_non_market_news_rejected():
    for text in [
        "Milli takım hazırlık maçında galip geldi",
        "Ünlü oyuncu ödül töreninde görüntülendi",
        "Yeni grip salgınına karşı aşı uyarısı",
    ]:
        assert not is_market_related(text), text
        assert market_category(text) is None


# --- Piyasa/ekonomi haberleri kabul edilmeli ---
def test_market_news_accepted():
    for text in [
        "Borsa İstanbul günü yükselişle kapattı",
        "TCMB faiz kararını açıkladı, enflasyon beklentisi değişti",
        "Altın fiyatları rekor kırdı, dolar/TL yatay seyrediyor",
        "Halka arz takvimi: yeni şirketler borsada işlem görecek",
    ]:
        assert is_market_related(text), text


# --- Sirket adi eslesmesi kodu gerektirmez ---
def test_company_name_match_without_code():
    assert match_ticker("Migros yeni mağaza açılışını duyurdu") == "MGROS.IS"
    assert match_text_for_ticker("Kervan Gıda ihracat rekoru kırdı", "KERVN.IS")
    assert match_text_for_ticker("Ahlatcı Doğal Gaz'ın hedef fiyatı yükseltildi", "AHGAZ.IS")


# --- Yeni eklenen hisseler eslesmeli ---
def test_new_basket_news_matching():
    assert match_text_for_ticker("Sanica Isı yeni sipariş aldı", "SNICA.IS")
    assert match_text_for_ticker("Obase büyüme hedefini açıkladı", "OBASE.IS")
    assert match_text_for_ticker("Kayseri Şeker temettü dağıtacak", "KAYSE.IS")


# --- Ticker kodu buyuk harf tam kelime olarak eslesmeli ---
def test_code_match_is_case_sensitive():
    assert match_ticker("(KAREL) pay sırası brüt takasa alındı") == "KAREL.IS"
    # Kucuk harfli genel kelime yanlis eslesmemeli
    assert match_ticker("istem için gerekli tüm bilgiler eklendi") is None


def test_filter_market_news_keeps_order():
    items = [
        _item("Borsa İstanbul rekor seviyeden kapandı"),
        _item("Ünlü şarkıcı konser verdi"),
        _item("Migros yeni şube açtı"),
    ]
    kept = filter_market_news(items)
    assert [k.headline for k in kept] == [
        "Borsa İstanbul rekor seviyeden kapandı",
        "Migros yeni şube açtı",
    ]
