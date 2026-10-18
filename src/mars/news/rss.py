# SPDX-License-Identifier: AGPL-3.0-only
"""RSS haber akislari — piyasa haberlerini ceker ve hisseyle eslestirir.

Tasarim:
  * Standart kutuphane (xml.etree) ile RSS 2.0 parse — agir bagimlilik yok.
  * HTTP icin requests + makul timeout; bir akis dusse digerleri calismaya devam
    eder (kismi basari kabul edilir).
  * Eslestirme: haber metninde sirket adi/ticker aramasi (tickers.py).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
import requests

from ..config import Settings, load_settings
from .sources import NewsItem
from .tickers import match_ticker, match_text_for_ticker

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MARS/0.1; research project)"}


@dataclass
class FeedResult:
    items: list[NewsItem]
    source: str
    ok: bool
    error: str = ""


def _parse_date(raw: str | None) -> pd.Timestamp | None:
    if not raw:
        return None
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return pd.Timestamp(dt).tz_localize(None)
    except Exception:  # noqa: BLE001
        return None


def parse_rss(xml_text: str | bytes, source: str = "rss") -> list[NewsItem]:
    """RSS 2.0 metnini NewsItem listesine cevirir (ag erisimi YOKTUR — test edilebilir).

    Bytes verilirse ET XML bildirimindeki kodlamayi kullanir (mojibake'i onler);
    str verilirse dogrudan parse eder.
    """
    root = ET.fromstring(xml_text)
    items: list[NewsItem] = []
    for node in root.iter("item"):
        title = (node.findtext("title") or "").strip()
        desc = (node.findtext("description") or "").strip()
        link = (node.findtext("link") or "").strip() or None
        pub = _parse_date(node.findtext("pubDate"))
        if not title:
            continue
        items.append(
            NewsItem(
                headline=title,
                body=desc,
                source=source,
                url=link,
                timestamp=pub,
            )
        )
    return items


def fetch_feed(url: str, source: str | None = None, timeout: int = 12) -> FeedResult:
    """Tek bir RSS akisini ceker ve parse eder."""
    if "//" in url:
        source = source or url.split("/")[2]
    else:
        source = source or url
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        # NOT: resp.text yerine content — bazi akislar charset bildirmez ve
        # requests latin-1'e duser; ET XML bildiriminden dogru cozer.
        items = parse_rss(resp.content, source=source)
        return FeedResult(items=items, source=source, ok=True)
    except Exception as exc:  # noqa: BLE001
        return FeedResult(items=[], source=source, ok=False, error=f"{type(exc).__name__}: {exc}")


def fetch_all_feeds(settings: Settings | None = None) -> tuple[list[NewsItem], list[str]]:
    """Tum yapilandirilmis akislari ceker. Donus: (haberler, hata listesi)."""
    settings = settings or load_settings()
    all_items: list[NewsItem] = []
    errors: list[str] = []
    for url in [u.strip() for u in settings.rss_feeds.split(",") if u.strip()]:
        res = fetch_feed(url)
        if res.ok:
            all_items.extend(res.items)
        else:
            errors.append(f"{res.source}: {res.error}")
    # Yeniden eskiye sirala, limit uygula
    dated = [i for i in all_items if i.timestamp is not None]
    dated.sort(key=lambda i: i.timestamp, reverse=True)
    limited = dated[: settings.rss_max_items]
    return limited, errors


def news_for_ticker(ticker: str, items: list[NewsItem]) -> list[NewsItem]:
    """Bir ticker ile eslesen haberleri dondurur (sirket adi bazli)."""
    return [i for i in items if match_text_for_ticker(i.text, ticker)]


def auto_match(items: list[NewsItem]) -> dict[str, list[NewsItem]]:
    """Haberleri ticker'lara otomatik baglar (screener icin)."""
    out: dict[str, list[NewsItem]] = {}
    for i in items:
        t = match_ticker(i.text)
        if t:
            out.setdefault(t, []).append(i)
    return out
