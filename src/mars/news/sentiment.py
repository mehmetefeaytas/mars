# SPDX-License-Identifier: AGPL-3.0-only
"""Turkce haber/duygu polaritesi.

Hibrit tasarim:
  1. BERT (birincil): BERTurk tabanli duygu modeli (varsa `transformers`+`torch`).
     Varsayilan: savasy/bert-base-turkish-sentiment-cased.
  2. Lexicon (fallback): Turkce finansal sozluk + olumsuzlama/bogus kurallari.
  3. LLM (opsiyonel): dusuk guvende yerel LLM'e dusme (bkz. llm.py).

Not: BERT cikisi modele gore etiket sirasi degisebilir; etiket->polarite eslemesi
esnek yapilir. Backend bulunamazsa sistem calismaya devam eder (lexicon).
"""

from __future__ import annotations

import re
from functools import lru_cache

from .sources import SentimentResult

_TURKISH_LOWER = str.maketrans({"İ": "i", "I": "ı", "Ş": "ş", "Ğ": "ğ", "Ü": "ü", "Ö": "ö", "Ç": "ç"})

POSITIVE_WORDS = {
    "artış", "artti", "arttı", "artis", "yükseliş", "yükseldi", "yükselişe", "kar",
    "kâr", "karlilik", "kârlılık", "rekor", "guclu", "güçlü", "büyüme", "büyüdü",
    "kazanç", "olumlu", "pozitif", "iyilesme", "iyileşme", "beklenti üstü", "sozlesme",
    "sözleşme", "ihale", "yatirim", "yatırım", "temettu", "temettü", "geri alim",
    "geri alım", "kapasite", "hedef yükseltti", "beğen", "tavsiye", "al", "prim",
    "toparlanma", "toparladı", "iyimser", "rally", "ralli", "kazandı", "güçlendi",
    "olumlu", "teşvik", "destek", "onay", "onaylandı", "anlaşma", "anlaştı",
}

NEGATIVE_WORDS = {
    "düşüş", "dusus", "düştü", "dustu", "zarar", "kayip", "kayıp", "gerileme",
    "geriledi", "zayif", "zayıf", "kuculme", "küçülme", "olumsuz", "negatif",
    "kotulesme", "kötüleşme", "beklenti alti", "beklenti altı", "iflas", "konkordato",
    "dava", "ceza", "soruşturma", "soruşturma", "manipulasyon", "manipülasyon",
    "borc", "borç", "temerrut", "temerrüt", "satis", "satış", "sat", "uyari", "uyarı",
    "risk", "düşürdü", "dusurdu", "indirim", "iptal", "durdurma", "durduruldu",
    "kotu", "kötü", "kriz", "endişe", "endişe", "panik", "çöküş", "cokus", "eridi",
    "daralma", "işten çıkarma", "üretim durdu",
}

NEGATIONS = {"degil", "değil", "yok", "yoktu", "olmadı", "olmadi", "hayir", "hayır", "yoktur"}
INTENSIFIERS = {"rekor", "sert", "buyuk", "büyük", "ciddi", "asiri", "aşırı", "çok", "cok"}

_TOKEN_RE = re.compile(r"[a-zçğıöşü]+")


def _normalize(text: str) -> str:
    return text.translate(_TURKISH_LOWER).lower()


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(_normalize(text))


@lru_cache(maxsize=1)
def _try_load_bert(model_name: str):
    """BERT duygu boru hattini yuklemeye calisir; yoksa None."""
    try:
        from transformers import pipeline  # lokal import

        return pipeline("sentiment-analysis", model=model_name)
    except Exception:  # noqa: BLE001
        return None


def _bert_sentiment(text: str, model_name: str) -> SentimentResult | None:
    pipe = _try_load_bert(model_name)
    if pipe is None:
        return None
    try:
        out = pipe(text[:512])[0]
    except Exception:  # noqa: BLE001
        return None
    label = str(out.get("label", "")).upper()
    score = float(out.get("score", 0.5))
    positive = ("POS" in label) or ("LABEL_1" in label) or ("OLUM" in label)
    polarity = score if positive else -score
    return SentimentResult(
        polarity=round(polarity, 4),
        label="pozitif" if positive else "negatif",
        confidence=round(abs(score), 4),
        backend="bert",
    )


def lexicon_sentiment(text: str) -> SentimentResult:
    """Turkce finansal sozluk tabanli polarite."""
    toks = _tokens(text)
    pos = neg = 0
    intens = 0
    negate_next = False
    negated_hits = 0

    for tok in toks:
        if tok in NEGATIONS:
            negate_next = True
            continue
        if tok in INTENSIFIERS:
            intens += 1
            continue
        if tok in POSITIVE_WORDS:
            if negate_next:
                neg += 1
                negated_hits += 1
            else:
                pos += 1
        elif tok in NEGATIVE_WORDS:
            if negate_next:
                pos += 1
                negated_hits += 1
            else:
                neg += 1
        negate_next = False

    total = pos + neg
    if total == 0:
        return SentimentResult(polarity=0.0, label="notr", confidence=0.2, backend="lexicon")

    polarity = (pos - neg) / total
    # Siddet: vurularin sayisina ve yogunlastiricilara gore guven.
    base_conf = min(1.0, total / 5.0)
    conf = min(1.0, base_conf + 0.1 * intens + 0.05 * negated_hits)
    label = "pozitif" if polarity > 0.05 else "negatif" if polarity < -0.05 else "notr"
    return SentimentResult(
        polarity=round(polarity, 4), label=label, confidence=round(conf, 4), backend="lexicon"
    )


def analyze_sentiment(
    text: str,
    *,
    backend: str = "auto",
    model_name: str = "savasy/bert-base-turkish-sentiment-cased",
) -> SentimentResult:
    """Hibrit duygu analizi. backend: auto | bert | lexicon."""
    if backend in {"auto", "bert"}:
        res = _bert_sentiment(text, model_name)
        if res is not None:
            return res
        if backend == "bert":
            # BERT zorunlu ama yuklenemedi -> lexicon'a dus (uyari niteliginde backend)
            fallback = lexicon_sentiment(text)
            fallback.backend = "lexicon(bert-unavailable)"
            return fallback
    return lexicon_sentiment(text)
