# SPDX-License-Identifier: AGPL-3.0-only
"""LLM cikti temizligi ve istem kurma testleri."""

from __future__ import annotations

from mars.news.llm import build_explanation_prompt, clean_completion


def test_strips_think_block():
    text = "<think>ic muhakeme</think>Sonuc: skor yüksek. Bu bir yatırım tavsiyesi değildir."
    out = clean_completion(text)
    assert "ic muhakeme" not in out
    assert "skor yüksek" in out


def test_strips_unclosed_think_block():
    text = "<think>uzun uzun dusunuyor ama kapanmadi"
    assert clean_completion(text) == ""


def test_removes_duplicate_sentences():
    text = "Hacim yüksek. Hacim yüksek. Fiyat arttı. Hacim yüksek."
    out = clean_completion(text)
    assert out.count("Hacim yüksek") == 1
    assert "Fiyat arttı" in out


def test_cuts_repetition_loop():
    # Kucuk modellerin tipik dongusu: ayni cumle onlarca kez
    text = " ".join(["Skor yüksek."] * 40)
    out = clean_completion(text)
    assert out.count("Skor yüksek") <= 3


def test_limits_sentence_count():
    text = " ".join(f"Cümle {i} burada." for i in range(50))
    out = clean_completion(text, max_sentences=5)
    assert len(out.split(".")) <= 7


def test_prompt_forbids_inventing_dates():
    payload = {
        "ticker": "KAREL.IS",
        "date": "2026-10-16",
        "suspicion": 66.0,
        "data_confidence": 85.0,
        "log_ret": "+9.5%",
        "abnormal_ret": "+9.2%",
        "liquidity_tier": "sığ",
        "reasons": [{"signal": "İşlem hacmi", "points": 18.0, "detail": "hacim 6 kat"}],
        "news": [],
    }
    prompt = build_explanation_prompt(payload)
    assert "KAREL.IS" in prompt
    assert "2026-10-16" in prompt
    assert "hacim 6 kat" in prompt
    assert "haber yok" in prompt.lower()
