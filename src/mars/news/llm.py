# SPDX-License-Identifier: AGPL-3.0-only
"""LLM kopru katmani — Ollama / EVREN / OpenAI-uyumlu saglayicilar.

Tasarim:
  * Tek bir OpenAI-uyumlu sohbet cagrisi (`POST {base}/chat/completions`).
  * Saglayiciya gore kimlik dogrulama: Ollama -> yok; EVREN -> X-API-Key +
    Bearer; OpenAI/custom -> Bearer.
  * EVREN gecidinin tek seferlik "kullanim sartlari" kapisi desteklenir
    (403 terms_not_accepted -> `LLMTermsRequired`).
  * Kucuk modellerde gorulen TEKRAR DONGUSU iki katmanda engellenir:
    (a) uretim parametreleri (frequency/presence penalty, top_p),
    (b) `clean_completion` ile cikti sonrasi temizlik.
  * LLM yoksa/hataliysa sistem kural-tabanli aciklamaya duser (None doner).
"""

from __future__ import annotations

import json
import re

import requests

from ..config import Settings, load_settings
from .sources import SentimentResult

# --------------------------------------------------------------------------
# Hatalar
# --------------------------------------------------------------------------
class LLMError(Exception):
    """LLM katmanindan gelen genel hata."""


class LLMTermsRequired(LLMError):
    """Saglayici kullanim sartlarinin onaylanmasini istiyor (EVREN kapisi)."""

    def __init__(self, version: int | None = None, message: str = ""):
        self.version = version
        super().__init__(message or "Kullanim sartlari onaylanmali.")


# --------------------------------------------------------------------------
# Kimlik dogrulama / istek yardimcilari
# --------------------------------------------------------------------------
def is_remote(settings: Settings) -> bool:
    return settings.llm_provider.lower() in {"evren", "openai", "custom"}


def auth_headers(settings: Settings) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = (settings.llm_api_key or "").strip()
    if not key:
        return headers
    headers["Authorization"] = f"Bearer {key}"
    if settings.llm_provider.lower() == "evren":
        headers["X-API-Key"] = key
    return headers


def _url(settings: Settings, path: str) -> str:
    return f"{settings.llm_base_url.rstrip('/')}/{path.lstrip('/')}"


def _extract_error(resp: requests.Response) -> tuple[str, int | None]:
    """(mesaj, gerekli_sart_surumu) dondurur."""
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001
        return (resp.text[:300] or f"HTTP {resp.status_code}", None)
    err = body.get("error") if isinstance(body, dict) else None
    text = ""
    version = None
    if isinstance(err, dict):
        text = str(err.get("message") or err.get("code") or "")
        evren = err.get("evren") if isinstance(err.get("evren"), dict) else {}
        version = evren.get("terms_version") or evren.get("accepted_version")
    elif isinstance(err, str):
        text = err
    if not text:
        text = json.dumps(body, ensure_ascii=False)[:300]
    return text, (int(version) if version else None)


# --------------------------------------------------------------------------
# Kesif uclari
# --------------------------------------------------------------------------
def llm_available(settings: Settings | None = None) -> bool:
    settings = settings or load_settings()
    if not settings.llm_enabled:
        return False
    if not is_remote(settings):
        # Yerel Ollama: anahtar gerekmez
        pass
    try:
        r = requests.get(_url(settings, "models"), headers=auth_headers(settings), timeout=3)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def fetch_models(settings: Settings | None = None) -> list[dict]:
    """Saglayicidan model listesi (id + meta) ceker."""
    settings = settings or load_settings()
    r = requests.get(_url(settings, "models"), headers=auth_headers(settings), timeout=8)
    r.raise_for_status()
    data = r.json()
    raw = data.get("data", data.get("models", [])) if isinstance(data, dict) else []
    out = []
    for m in raw:
        if isinstance(m, dict):
            out.append({"id": m.get("id") or m.get("name"), "owned_by": m.get("owned_by"),
                        "task": m.get("task"), "modalities": m.get("modalities")})
        elif isinstance(m, str):
            out.append({"id": m})
    return [m for m in out if m.get("id")]


def terms_status(settings: Settings | None = None) -> dict:
    settings = settings or load_settings()
    r = requests.get(_url(settings, "terms/status"), headers=auth_headers(settings), timeout=8)
    if r.status_code == 404:
        return {"supported": False, "accepted": True}
    r.raise_for_status()
    body = r.json()
    body["supported"] = True
    return body


def accept_terms(settings: Settings | None = None, version: int = 1) -> dict:
    settings = settings or load_settings()
    r = requests.post(
        _url(settings, "terms/accept"),
        headers=auth_headers(settings),
        json={"version": version},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------
# Sohbet cagrisi
# --------------------------------------------------------------------------
def chat(
    messages: list[dict],
    settings: Settings | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> str:
    """OpenAI-uyumlu sohbet tamamlama. Ham metni dondurur."""
    settings = settings or load_settings()
    temp = settings.llm_temperature if temperature is None else temperature
    tokens = settings.llm_max_tokens if max_tokens is None else max_tokens
    if is_remote(settings) and tokens < 1024:
        tokens = 1024

    body: dict = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": temp,
        "max_tokens": tokens,
        # Tekrar dongusunu bastir (Ollama ve OpenAI-uyumlu uclarda destekli)
        "frequency_penalty": 0.4,
        "presence_penalty": 0.3,
        "top_p": 0.9,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if is_remote(settings) and not settings.llm_thinking:
        # EVREN'e ozgu uzanti: muhakemeyi kapat (sohbet modellerinde yok sayilir)
        body["thinking"] = False

    try:
        r = requests.post(
            _url(settings, "chat/completions"),
            headers=auth_headers(settings),
            json=body,
            timeout=settings.llm_timeout,
        )
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"Baglanti hatasi: {exc}") from exc

    if r.status_code == 403:
        msg, version = _extract_error(r)
        if "terms" in msg.lower() or version is not None:
            raise LLMTermsRequired(version=version, message=msg)
        raise LLMError(msg)
    if r.status_code >= 400:
        msg, _ = _extract_error(r)
        raise LLMError(msg)

    try:
        payload = r.json()
        msg_obj = payload["choices"][0]["message"]
        content = msg_obj.get("content") or ""
        # EVREN/Dusunen modeller: sonuc bazen content yerine reasoning'de kalabilir
        if not str(content).strip():
            content = msg_obj.get("reasoning_content") or msg_obj.get("reasoning") or ""
        return str(content)
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"Yanit cozumlenemedi: {exc}") from exc


# --------------------------------------------------------------------------
# Cikti temizligi (tekrar dongusu kirici)
# --------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think[^>]*>.*", re.DOTALL | re.IGNORECASE)


def clean_completion(text: str, max_sentences: int = 12) -> str:
    """LLM ciktisini temizler: muhakeme blogu, bos tekrarlar ve donguler atilir."""
    if not text:
        return ""
    out = _THINK_RE.sub(" ", text)
    out = _THINK_OPEN_RE.sub(" ", out)
    out = out.replace("</think>", " ").replace(r"<\/think>", " ")
    out = re.sub(r"[ \t]+", " ", out)

    # Satirlara ayir, tekrar eden satirlari at
    lines = [ln.strip(" -•\t") for ln in out.splitlines()]
    seen_lines: set[str] = set()
    kept_lines: list[str] = []
    for ln in lines:
        if not ln:
            continue
        key = re.sub(r"\W+", " ", ln.lower()).strip()
        if key and key in seen_lines:
            continue
        if key:
            seen_lines.add(key)
        kept_lines.append(ln)

    text2 = " ".join(kept_lines)
    text2 = re.sub(r"\s+", " ", text2).strip()

    # Cumlelere ayir; birebir tekrar eden cumleleri at
    sentences = re.split(r"(?<=[.!?…])\s+", text2)
    seen: set[str] = set()
    kept: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        key = re.sub(r"\W+", " ", s.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        kept.append(s)
    kept = _cut_ngram_loop(kept)
    if len(kept) > max_sentences:
        kept = kept[:max_sentences]
    return " ".join(kept).strip()


def _cut_ngram_loop(sentences: list[str], max_repeat: int = 2) -> list[str]:
    """Ayni cumle kalibi ust uste max_repeat'ten fazla tekrar ediyorsa kes."""
    out: list[str] = []
    window: dict[str, int] = {}
    prev_key = None
    run = 0
    for s in sentences:
        key = re.sub(r"\W+", " ", s.lower()).strip()
        if key and key == prev_key:
            run += 1
            if run >= max_repeat:
                continue
        else:
            run = 0
        prev_key = key
        # Genel tekrar sayimi: ayni cumle toplamda 3'ten fazla gecmesin
        window[key] = window.get(key, 0) + 1
        if window[key] > 3:
            continue
        out.append(s)
    return out


# --------------------------------------------------------------------------
# Duygu analizi (haber polaritesi)
# --------------------------------------------------------------------------
_SENTIMENT_SYSTEM = (
    "Sen bir finansal haber analistisin. Verilen Turkce haber basligini/ozetini "
    "incele ve SADECE su JSON semasini dondur: "
    '{"polarity": <float -1..1>, "label": "pozitif|negatif|notr", "confidence": <float 0..1>}. '
    "Baska aciklama yazma."
)


def llm_sentiment(text: str, settings: Settings | None = None) -> SentimentResult | None:
    """LLM ile polarite. Basarisizsa None (cagiran fallback yapar)."""
    settings = settings or load_settings()
    if not settings.llm_enabled:
        return None
    try:
        content = chat(
            [
                {"role": "system", "content": _SENTIMENT_SYSTEM},
                {"role": "user", "content": text[:2000]},
            ],
            settings,
            temperature=0.0,
            max_tokens=200,
            json_mode=True,
        )
        data = json.loads(content)
        return SentimentResult(
            polarity=float(data.get("polarity", 0.0)),
            label=str(data.get("label", "notr")),
            confidence=float(data.get("confidence", 0.5)),
            backend="llm",
        )
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# Gun aciklamasi
# --------------------------------------------------------------------------
_EXPLAIN_SYSTEM = (
    "Sen MARS adli borsa izleme programinin yardimci asistanisin. Okuyucu borsa "
    "konusunda tamamen yenidir; hicbir teknik bilgisi yokmus gibi yaz.\n"
    "Kurallar:\n"
    "1) Yalnizca Turkce yaz.\n"
    "2) Basit, gunluk konusma dili kullan. Teknik bir terim kullanman gerekirse "
    "ayni cumlede parantez icinde tek cumleyle acikla.\n"
    "3) Kisa cumleler kur. Madde isareti, baslik veya kod kullanma. En fazla 150 kelime.\n"
    "4) Yalnizca sana verilen sayilari kullan. VERILMEYEN hicbir tarih, rakam, haber "
    "veya olay uydurma. Tarihleri tekrar tekrar yazma.\n"
    "5) Ayni cumleyi ya da benzerini asla tekrarlama.\n"
    "6) 'Kesinlikle manipule edildi' gibi kesin hukum verme; 'supheli', 'dikkat cekici', "
    "'normalin disinda' gibi ihtiyatli ifadeler kullan.\n"
    "7) Sirasiyla: (a) bir cumlede o gun ne oldugunu soyle, (b) skoru yukselten 1-2 "
    "nedeni sayiyla acikla, (c) haber varsa etkisini bir cumleyle belirt, (d) "
    "'Bu bir yatirim tavsiyesi degildir.' diye bitir.\n"
    "8) Duz paragraf yaz; JSON, tablo veya kod blogu yazma."
)


def build_explanation_prompt(payload: dict) -> str:
    """LLM'e verilecek kullanici istemini kurar (payload: gun verisi + haberler)."""
    lines = [
        f"Hisse: {payload.get('ticker')}",
        f"Tarih: {payload.get('date')}",
        f"Suphe skoru: {payload.get('suspicion')}/100",
        f"Veri guveni: {payload.get('data_confidence')}/100",
        f"Gunluk getiri: {payload.get('log_ret')}",
        f"Anormal getiri (endekse gore fazlasi): {payload.get('abnormal_ret')}",
        f"Likidite katmani: {payload.get('liquidity_tier')}",
        "Skoru yukselten sinyaller (puan):",
    ]
    for r in payload.get("reasons", []):
        lines.append(f"- {r.get('signal')}: +{r.get('points')} puan — {r.get('detail')}")
    news = payload.get("news", [])
    if news:
        lines.append("Ayni gun cikan haberler:")
        for n in news:
            lines.append(
                f"- [{n.get('source')}] {n.get('headline')} "
                f"(duygu {n.get('polarity'):+.2f}, guvenilirlik {n.get('credibility'):.2f})"
            )
    else:
        lines.append("Ayni gun hisseyle eslesen dogrulanmis haber yok.")
    lines.append("Yukaridaki kurallara gore bu gunu basit dille acikla.")
    return "\n".join(lines)


def explain_with_llm(payload: dict, settings: Settings | None = None) -> tuple[str | None, str | None]:
    """Yerel/uzak LLM ile gunun aciklamasi.

    Returns: (metin, hata_mesaji). Basarili ise (metin, None); basarisizsa
    (None, "aciklama") ve cagiran kural-tabanli fallback yapar.
    """
    settings = settings or load_settings()
    if not settings.llm_enabled:
        return None, "LLM kapali."
    try:
        raw = chat(
            [
                {"role": "system", "content": _EXPLAIN_SYSTEM},
                {"role": "user", "content": build_explanation_prompt(payload)},
            ],
            settings,
            temperature=0.3,
        )
        cleaned = clean_completion(raw)
        return (cleaned or None), (None if cleaned else "LLM bos yanit dondurdu.")
    except LLMTermsRequired as exc:
        raise
    except LLMError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
