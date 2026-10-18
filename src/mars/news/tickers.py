# SPDX-License-Identifier: AGPL-3.0-only
"""BIST ticker -> sirket adi eslemeleri (haber eslestirmesi icin).

Haber basligindan hisse bulmak icin kullanilir; eslesme Turkce/kucuk-harf
normalizasyonu ile alt-dize aramasi uzerinden yapilir.
"""

from __future__ import annotations

TICKER_NAMES: dict[str, list[str]] = {
    "THYAO.IS": ["türk hava yolları", "türk hava yollari", "thy"],
    "GARAN.IS": ["garanti bbva", "garanti bankası", "garanti bankasi", "garanti"],
    "AKBNK.IS": ["akbank"],
    "ISCTR.IS": ["iş bankası", "is bankası", "iş bankasi", "is bankasi"],
    "YKBNK.IS": ["yapı kredi", "yapi kredi"],
    "HALKB.IS": ["halkbank"],
    "VAKBN.IS": ["vakıfbank", "vakifbank"],
    "ASELS.IS": ["aselsan"],
    "EREGL.IS": ["erdoğan demir", "erdemir", "ereğli demir"],
    "SISE.IS": ["şişecam", "sisecam"],
    "KCHOL.IS": ["koç holding", "koç topluluğu", "koç grup"],
    "SAHOL.IS": ["sabancı holding", "sabancı topluluğu"],
    "BIMAS.IS": ["bim", "bİm mağazalar"],
    "TUPRS.IS": ["tüpraş", "tupras"],
    "PETKM.IS": ["petkim"],
    "SASA.IS": ["sasa polyester", "sasa"],
    "FROTO.IS": ["ford otosan", "otosan"],
    "TOASO.IS": ["tofaş", "tofas"],
    "KRDMD.IS": ["kardemir"],
    "OYAKC.IS": ["oyak çimento", "oyak cimento"],
    "ARCLK.IS": ["arçelik", "arcelik"],
    "KONTR.IS": ["kontroldenetim", "kontrol denetim"],
    "ASTOR.IS": ["astor enerji", "astor"],
    "MGROS.IS": ["migros"],
    "SOKM.IS": ["şok marketler", "sok marketler"],
    "EKGYO.IS": ["eki konut", "eki gayrimenkul"],
    "TSKB.IS": ["kalkınma bankası", "tskb"],
    "TCELL.IS": ["turkcell"],
    "TKFEN.IS": ["tekfen"],
    "ENKAI.IS": ["enkai", "enka"],
    "DOAS.IS": ["oto asistan", "doğuş otomotiv"],
    "ALARK.IS": ["alarko"],
    "DSTKF.IS": ["destek reklam"],
    "SMART.IS": ["smart güneş", "smart gunes"],
    "KAREL.IS": ["karel"],
    "VESTL.IS": ["vestel"],
    "BRSAN.IS": ["borusan birlik", "borusan"],
}

# Ters esleme: alt-dize -> ticker (ilk bulunan doner; uzun parcalar once denenir)
_REVERSE: list[tuple[str, str]] = sorted(
    ((frag.lower(), t) for t, frags in TICKER_NAMES.items() for frag in frags),
    key=lambda kv: -len(kv[0]),
)


def normalize_tr(text: str) -> str:
    return (
        text.replace("İ", "i").replace("I", "ı").replace("Ş", "ş").replace("Ğ", "ğ")
        .replace("Ü", "ü").replace("Ö", "ö").replace("Ç", "ç").replace("Â", "a")
        .replace("â", "a").lower()
    )


def match_ticker(text: str) -> str | None:
    """Bir haber metninden BIST ticker'i bulmaya calisir."""
    if not text:
        return None
    norm = normalize_tr(text)
    for frag, ticker in _REVERSE:
        if frag in norm:
            return ticker
    return None


def company_fragments(ticker: str) -> list[str]:
    return TICKER_NAMES.get(ticker.upper(), [])


def match_text_for_ticker(text: str, ticker: str) -> bool:
    """Metin verilen ticker'in sirket adiyla eslesiyor mu?"""
    frags = company_fragments(ticker)
    if not frags:
        return False
    norm = normalize_tr(text)
    return any(normalize_tr(f) in norm for f in frags)
