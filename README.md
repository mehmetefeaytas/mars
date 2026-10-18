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
   fallback → opsiyonel **yerel LLM** (Ollama) fallback.
3. **RSS akışları:** BloombergHT ve CNN Türk Ekonomi canlı çekilir; haberler
   BIST şirket adlarıyla otomatik eşleştirilir (37 ticker eşlemesi).

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

## Terminal Arayüzü

```bash
PYTHONPATH=src .venv/bin/streamlit run app.py
```

| Sekme | İçerik |
|-------|--------|
| 🔍 Şüpheli Hisseler | BIST sepeti taraması; skor sıralaması, veri güveni, getiri/AR, en etkili sinyal, CSV indirme |
| 📊 Hisse Analizi | Mum + hacim + şüphe grafiği (Plotly), KPI kartları, sinyal şelalesi, **değer bazlı gerekçeler** (robust z, AR), gün seçici |
| 📰 Haber Akışı | RSS haberleri; polarite ve VeriShield güvenilirlik rozetleri, hisse eşleştirme |
| 🤖 LLM Açıklama | Yerel Ollama modeliyle günün yorumu; kural-tabanlı açıklama her zaman görünür |

Kenar çubuğunda 10+ parametre: geçmiş uzunluğu, robust pencere, market model
penceresi, iForest ayarları, 7 füzyon ağırlığı, haber ceza gücü, RSS/LLM anahtarları.
LLM için: `ollama serve && ollama pull qwen2.5:7b-instruct`

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
  anomaly/         robust_stats (MCD), iforest, evt, changepoint
  models/          market_model (event study / CAR)
  news/            sentiment (BERT+lexicon), llm (+açıklama), rss, tickers, verishield_adapter
  fusion/          weighted (kısıtlı, gün-bazlı normalize), evidence (D-S), calibration
  ml.py            denetimli sınıflandırıcı (eğit/yükle/tahmin)
  explain.py       katkı şelalesi + değer bazlı gerekçeler + doğal dil özeti
  pipeline.py      uçtan uca orkestrasyon
  experiments.py   baseline vs MARS, ablation, stres testi
app.py             Streamlit terminal (4 sekme, trading-terminal teması)
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
