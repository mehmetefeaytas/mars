# MARS — BIST Piyasa Anomali ve Açıklanabilir Risk Skorlama Sistemi

> **Fark et → Skorla → Açıkla**
> Fiyat, hacim, oynaklık ve BIST 100 ayrışmasını çok boyutlu, robust istatistik ve
> makine öğrenmesi ile birleştiren; her skoru bileşenlerine ayırarak açıklayan bir
> **karar-destek** sistemidir.

MARS; hukuki hüküm vermez, yatırım tavsiyesi değildir. Anomali şüphe skoru bir
**olasılık**tır; karar her zaman insandadır.

## Neden MARS?

Tek bir fiyat sıçraması manipülasyon kanıtı değildir. Meşru bir şirket haberi veya
sığ tahtada düşük likidite de fiyatı fırlatır. Tekil eşik kuralları (örn. yalnızca
RSI/MACD veya hacim > eşik) yüksek oranda yanlış pozitif üretir. MARS, sinyalleri
vektör uzayında birleştirir, anomaliyi bileşenlerine ayırır ve **hangi sinyalin
skoru ne kadar artırdığını** yüzdesel olarak gösterir.

## Matematiksel Çekirdek (akademik kanıtlı)

| Rol | Yöntem | Kaynak |
|-----|--------|--------|
| Robust normalizasyon | Medyan + MAD | Rousseeuw & Hubert (2011) |
| Çok değişkenli aykırılık | Robust Mahalanobis (MCD) | Rousseeuw (1984) |
| Aykırılık (ML) | Isolation Forest | Liu, Ting, Zhou (2008) |
| Endeks ayrışması | Event Study / Market Model (AR, CAR) | MacKinlay (1997) |
| Füzyon | Kısıtlı ağırlıklı (Σw=1, w≥0) + Dempster-Shafer | Shafer (1976) |
| Kuyruk eşiği (ileri) | EVT / Peaks-Over-Threshold (GPD) | McNeil & Frey (2000) |
| Rejim kırılması (ileri) | Bayesian Online Change-Point Detection | Adams & MacKay (2007) |
| Açıklanabilirlik | Şelale katkı + SHAP (opsiyonel) | Ribeiro (2016), Lundberg & Lee (2017) |

Ayrıntılar: [`academic.md`](academic.md). Deney sonuçları: [`docs/results.md`](docs/results.md).

## Haber & Güvenilirlik Entegrasyonu

Haber sinyali iki eksende modellenir:

1. **Güvenilirlik (0–1):** [VeriShield](https://github.com/mehmetefeaytas/verishield-core)
   motoru (6 kanal: metin, görsel, mekân-zaman, çapraz doğrulama, koordinasyon,
   resmi kaynak) **gömülü kütüphane** olarak kullanılır (git submodule, AGPL-3.0).
2. **Polarite & şiddet:** Türkçe **BERT** duygu modeli (birincil) → **lexicon**
   fallback → opsiyonel **LLM** fallback.
3. **RSS akışları:** BloombergHT ve CNN Türk Ekonomi canlı çekilir; haberler
   BIST şirket adlarıyla otomatik eşleştirilir (52 hisse eşlemesi).
4. **Borsa dışı filtre:** magazin, spor, sağlık ve genel siyaset haberleri
   elenir; yalnızca piyasa/ekonomi ve şirket haberleri analize girer
   (`news/filters.py`). Bir haberde **şirket adının geçmesi yeterlidir**,
   ticker kodu şart değildir.

Etkin haber değeri: `news_value = polarite × güvenilirlik`. Bayesci sezgiyle,
doğrulanmış ve hareketle **aynı yönde** bir haber şüpheyi düşürür; **çelişen**
güvenilir haber hafifçe artırır; düşük güvenilirlikli söylenti nötr kalır.
VeriShield bulunamazsa yerel sezgisel bir güvenilirlik skorlayıcı devreye girer;
sistem yine çalışır.

## Denetimli ML Katmanı

Etiket şeması: `pump`/`subtle` → şüpheli, `news_legit` → meşru (kontrollü sentetik).
Model: imputer + scaler + LogisticRegression. **Holdout AUC 0.9939**; model
`data/models/mars_ml.joblib` altında kayıtlı ve `sig_ml` sinyali olarak füzyona
girer. Eğitim:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_ml.py --seeds 30
```

Colab'da eğitim ve **gerçek vaka (`cases.csv`: ticker,tarih,etiket) ekleme**:
`notebooks/mars_ml_colab.ipynb` — SPK kararlarından derlediğiniz etiketli günlerle
zenginleştirilebilir.

## Terminal Arayüzü (React + FastAPI)

**Birincil arayüz** — Figma trading-dashboard tasarım diliyle (koyu tema, sol
sidebar, veri-yoğun tablolar, kartlar) React + TypeScript + Vite üzerine kuruludur.

```bash
# 1) Backend (API + SPA, http://localhost:8010)
PYTHONPATH=src .venv/bin/uvicorn mars.api:app --port 8010

# 2) Gelistirme modu (hot-reload, http://localhost:5173)
cd frontend && npm install && npm run dev

# 3) Uretim build (frontend/dist -> API uzerinden servis edilir)
cd frontend && npm run build
```

| Sayfa | İçerik |
|-------|--------|
| 🔍 Şüpheli Hisseler | 52 hisselik sepet taraması; sıralanabilir skor tablosu, tahta bağlamı rozeti, CSV |
| 📊 Hisse Analizi | TradingView `lightweight-charts` mum+hacim ve şüphe grafikleri (senkron, tıklanabilir gün seçimi), 6 KPI kartı, sinyal şelalesi, sade gerekçeler, eşleşen haberler (yoksa genel piyasa haberleri) |
| 📰 Haber Akışı | Filtrelenmiş RSS kartları; kategori (hisse/piyasa/şirket) ve polarite/güvenilirlik rozetleri |
| 🤖 LLM Açıklama | Gün seçimi + LLM yorumu (başlangıç seviyesine uygun dil); LLM yoksa kural-tabanlı gerekçeler |
| ⚙️ Ayarlar | LLM sağlayıcı/model/API anahtarı/parametreler, bağlantı testi, şartlar onayı, RSS ayarları (`data/settings.local.json`) |
| ⚙️ Parametreler | 7 motor parametresi + 9 füzyon ağırlığı (canlı) — varsayılanlar önerilir |

Koyu/açık tema anahtarı üst çubuktan değiştirilir (tercih `localStorage`'da saklanır).

API uçları: `GET /api/health|meta|analyze|news` · `POST /api/screener|explain` ·
`GET|PUT /api/settings` · `GET /api/settings/models|terms` · `POST /api/settings/llm/test|terms/accept`.

### LLM Sağlayıcı Yapılandırması

LLM yorumu üç şekilde çalışır; hiçbiri zorunlu değildir (kural-tabanlı gerekçeler
her zaman üretilir). Arayüzden **⚙️ Ayarlar** sayfasından seçilir, `.env` ile de
verilebilir.

| Sağlayıcı | Base URL | Not |
|-----------|----------|-----|
| `ollama` | `http://localhost:11434/v1` | Yerel, internet gerekmez (örn. `qwen2.5:3b-instruct`) |
| `evren` | `https://evren-llmapi.ssyz.org.tr/v1` | EVREN LLM Gateway; `X-API-Key` + Bearer, kullanım şartları onayı gerekir |
| `openai` | `https://api.openai.com/v1` | OpenAI uyumlu |
| `custom` | elle | Herhangi bir OpenAI-uyumlu uç |

Tekrarlı/döngülü çıktıya karşı `frequency_penalty`, `presence_penalty`, `top_p` ve
çıktı temizliği (`clean_completion`: `think` blokları, yinelenen cümleler, n-gram
döngü kesme) uygulanır. Uç nokta ayarları `.env` → `data/settings.local.json`
sırasıyla geçersiz kılar (anahtar git'e girmez).

**Alternatif (hızlı demo):** `PYTHONPATH=src .venv/bin/streamlit run app.py`

## Manipülasyon Tespiti v3 — Likidite Bağlamı (Tahta Riski)

Akademik bulgu: manipülasyon düşük likiditeli hisselerde yoğunlaşır
(Aggarwal & Wu 2006; Khwaja & Mian 2005). Buna göre:

- **Tahta önseli:** medyan dolar hacimden likidite katmanı ("çok sığ" → "çok
  likit") ve `tahta_prior ∈ [0.15, 1.0]` hesaplanır.
- **Sınırlı modülasyon (veto değil):** derin/mega likit hissede hacim ve Amihud
  sinyal ağırlıkları aşırılır; kalan ağırlık diğer sinyallere dağıtılır. Örnek:
  BIMAS (likit) skoru 53.8 → 41.7'ye düştü, sığ tahtalar tam ağırlık korudu.
- **Yeni etiketler:** CLV/kapanış konumu (marking-the-close izi), çok günlük
  pump deseni (CAR₅ kalıcılığı + hacim sürekliliği + yön serisi), aralık oynaklığı.
- **ML yeniden eğitildi:** 8 özellik, holdout AUC 0.993.

## Kurulum

```bash
# Python 3.12+ ve uv (veya venv)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[all]"   # veya: pip install -r requirements.txt
cp .env.example .env
```

VeriShield gömülü modu için submodule'ü çekin:

```bash
git submodule update --init --depth 1
```

## Kullanım

### CLI

```bash
PYTHONPATH=src .venv/bin/python -m mars.cli THYAO.IS
PYTHONPATH=src .venv/bin/python -m mars.cli THYAO.IS --json
```

### Arayüz

```bash
PYTHONPATH=src .venv/bin/streamlit run app.py
```

### Deneyleri yeniden üret

```bash
PYTHONPATH=src .venv/bin/python scripts/run_experiments.py   # docs/results.md
```

### Test

```bash
PYTHONPATH=src .venv/bin/python -m pytest
```

## Mimari

```
Veri alımı → Temizleme/ehizalama → Özellik müh. → Anomali tespiti → Hibrit füzyon → Açıklama/UI
  yfinance      takvim hizalama     log getiri     MCD-Mahalanobis      kısıtlı ağırlık    şelale + kart
  OHLCV+BIST100                     MAD z          iForest              + D-S              katkı payları
                                    Amihud         market-model CAR     + haber düzeltmesi
```

```
src/mars/
  ingestion.py     yfinance .IS + BIST100 + kapsam-farkındalıklı Parquet cache + takvim
  features.py      log getiri, MAD z, aralık, Amihud
  signals.py       normalize [0,1] sinyal montajı (+ sig_ml)
  anomaly/         robust_stats (MCD), iforest, evt, changepoint, tahta
  models/          market_model (event study / CAR)
  news/            sentiment (BERT+lexicon), llm (çok sağlayıcılı), rss, tickers, filters
  fusion/          weighted (kısıtlı, gün-bazlı normalize), evidence (D-S), calibration
  ml.py            denetimli sınıflandırıcı (eğit/yükle/tahmin)
  explain.py       katkı şelalesi + sade gerekçeler + doğal dil özeti
  settings_store.py  arayüzden düzenlenebilen ayarlar (data/settings.local.json)
  api.py           FastAPI uçları + SPA servisi (frontend/dist)
  pipeline.py      uçtan uca orkestrasyon
  experiments.py   baseline vs MARS, ablation, stres testi
frontend/          React 18 + TS + Vite terminal arayüzü (6 sayfa, koyu/açık tema)
app.py             Streamlit alternatifi (hızlı demo)
notebooks/         Colab ML eğitim defteri
scripts/           train_ml.py, run_experiments.py
```

## Dürüst Sınırlar

- Etiketli zemin gerçeği (ground truth) yoktur; bu yüzden **uydurma F1/doğruluk
  metrikleri iddia edilmez**. Değerlendirme sentetik senaryolar ve denetimsiz
  metriklerle yapılır.
- Derinlik/emir defteri verisi MVP kapsamı dışıdır (lisanslı).
- yfinance lisans/erişim riskine karşı yerel Parquet önbelleği ve örnek arşiv vardır.
- Anomali skoru ≠ manipülasyon kanıtı.

## Lisans

AGPL-3.0-only. VeriShield gömülü bileşeni de AGPL-3.0'dır (telif: Mehmet Efe Aytaş).
