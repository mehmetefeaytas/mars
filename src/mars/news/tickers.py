# SPDX-License-Identifier: AGPL-3.0-only
"""BIST ticker -> sirket adi eslemeleri (haber eslestirmesi icin).

Iki katmanli eslestirme yapilir:
  1) Sirket adi parcalari (Turkce/kucuk-harf normalizasyonu + alt-dize aramasi).
  2) Ticker kodunun kendisi — metinde BUYUK harfle tam kelime olarak gecerse
     (KAP/piyasa haberlerinde sik gorulur). Kucuk harfli "info" gibi genel
     kelimelerin yanlis eslesmesini onlemek icin buyuk/kucuk harf duyarlidir.
"""

from __future__ import annotations

import re

TICKER_NAMES: dict[str, list[str]] = {
    # --- Banka / holding / buyuk olcekli ---
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
    "ARCLK.IS": ["arçelik", "arcelik"],
    "VESTL.IS": ["vestel"],
    "MGROS.IS": ["migros"],
    "SOKM.IS": ["şok marketler", "sok marketler"],
    "BRSAN.IS": ["borusan birlik", "borusan"],
    "KRDMD.IS": ["kardemir"],
    "OYAKC.IS": ["oyak çimento", "oyak cimento"],
    "KONTR.IS": ["kontroldenetim", "kontrol denetim"],
    "ASTOR.IS": ["astor enerji", "astor"],
    "EKGYO.IS": ["eki konut", "eki gayrimenkul"],
    "TSKB.IS": ["kalkınma bankası", "tskb"],
    "TCELL.IS": ["turkcell"],
    "TKFEN.IS": ["tekfen"],
    "ENKAI.IS": ["enkai", "enka"],
    "DOAS.IS": ["oto asistan", "doğuş otomotiv"],
    "ALARK.IS": ["alarko"],
    "SMART.IS": ["smart güneş", "smart gunes"],
    "KAREL.IS": ["karel"],
    # --- v3 ile sepete eklenen hisseler ---
    "ANELE.IS": ["anel elektrik"],
    "BTCIM.IS": ["batısöke", "batisoke", "batısöke söke"] ,
    "BURCE.IS": ["burçelik", "burcelik"],
    "CELHA.IS": ["çelik halat", "celik halat"],
    "EDATA.IS": ["e-data", "edata"],
    "GRTHO.IS": ["gürteks", "gurteks"],
    "GUNDG.IS": ["gündoğdu gıda", "gundogdu gida", "gündoğdu"],
    "HEDEF.IS": ["hedef holding", "hedef girişim", "hedef girisim"],
    "IEYHO.IS": ["ışıklar enerji", "isiklar enerji", "ışıklar holding"],
    "INFO.IS": ["info yatırım", "info yatirim"],
    "KERVN.IS": ["kervan gıda", "kervan gida", "kervan"],
    "KTLEV.IS": ["katılımevim", "katilimevim"],
    "KUYAS.IS": ["kuyaş yatırım", "kuyas yatirim", "kuyaş"],
    "MANAS.IS": ["manas enerji"],
    "ODINE.IS": ["odine"],
    "OZATD.IS": ["ozata"],
    "OZSUB.IS": ["özsu balık", "ozsu balik", "özsu"],
    "PASEU.IS": ["pasifik eurasia", "pasifik avrasya"],
    "PEKGY.IS": ["peker gayrimenkul", "peker gyo"],
    "PRZMA.IS": ["prizma"],
    "SELEC.IS": ["selçuk ecza", "selcuk ecza"],
    "AHGAZ.IS": ["ahlatcı doğal gaz", "ahlatci dogal gaz", "ahlatçı", "ahlatci"],
    "TERA.IS": ["tera yatırım", "tera yatirim"],
    "BVSAN.IS": ["bülbüloğlu", "bulbuloglu", "bülbüloğlu vinç"],
    "A1CAP.IS": ["a1 capital", "a1capital"],
    "SNICA.IS": ["sanica"],
    "KAYSE.IS": ["kayseri şeker", "kayseri seker"],
    "OBASE.IS": ["obase"],
    "SOKE.IS": ["söke değirmencilik", "soke degirmencilik"],
    "EYGYO.IS": ["eyüp gayrimenkul", "eyup gayrimenkul"],
    "DSTKF.IS": ["destek reklam", "destek faktoring"],
}

# Ters esleme: alt-dize -> ticker (uzun parcalar once denenir)
_REVERSE: list[tuple[str, str]] = sorted(
    ((frag.lower(), t) for t, frags in TICKER_NAMES.items() for frag in frags),
    key=lambda kv: -len(kv[0]),
)

# Kod eslestirmesi: en az 4 karakterli ticker kodlari (buyuk/kucuk harf duyarli)
_CODES: list[tuple[str, str]] = sorted(
    ((t.split(".")[0], t) for t in TICKER_NAMES if len(t.split(".")[0]) >= 4),
    key=lambda kv: -len(kv[0]),
)
_CODE_RE: dict[str, re.Pattern] = {
    code: re.compile(rf"(?<![A-Z0-9]){re.escape(code)}(?![A-Z0-9])")
    for code, _ in _CODES
}


def normalize_tr(text: str) -> str:
    return (
        text.replace("İ", "i").replace("I", "ı").replace("Ş", "ş").replace("Ğ", "ğ")
        .replace("Ü", "ü").replace("Ö", "ö").replace("Ç", "ç").replace("Â", "a")
        .replace("â", "a").lower()
    )


def _match_code(text: str) -> str | None:
    """Metinde BUYUK harfli ticker kodu ara (or. 'KAREL' veya '(KERVT)').
    Tamamen kucuk harfli metinde eslesme yapmaz."""
    for code, ticker in _CODES:
        if _CODE_RE[code].search(text):
            return ticker
    return None


def match_ticker(text: str) -> str | None:
    """Bir haber metninden BIST ticker'i bulmaya calisir."""
    if not text:
        return None
    norm = normalize_tr(text)
    for frag, ticker in _REVERSE:
        if frag in norm:
            return ticker
    return _match_code(text)


def company_fragments(ticker: str) -> list[str]:
    return TICKER_NAMES.get(ticker.upper(), [])


def match_text_for_ticker(text: str, ticker: str) -> bool:
    """Metin verilen ticker ile eslesiyor mu (sirket adi VEYA kod)?"""
    ticker = ticker.upper()
    frags = company_fragments(ticker)
    norm = normalize_tr(text)
    if frags and any(normalize_tr(f) in norm for f in frags):
        return True
    code = ticker.split(".")[0]
    if len(code) >= 4:
        return bool(_CODE_RE[code].search(text))
    return False
