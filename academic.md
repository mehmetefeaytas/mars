# MARS — Akademik Notlar

> Bu dosya, MARS projesinin akademik makaleye donusme surecindeki notlarini toplar.
> Her yontem birincil kaynaga dayandirilir; denklemler, tasarim kararlari ve
> deneysel bulgular ilerledikce eklenir.

## 1. Problemin Konumlandirilmasi

Finansal piyasalarda tekil fiyat sicramalari manipulasyon kaniti degildir; mesru
haber akisi veya likidite degisimleri de benzer dalgalanmalar uretir. MARS, BIST
pay piyasasinda fiyat, hacim, oynaklik ve endeks ayrismasi gibi cok boyutlu
sinyalleri birlestirerek **aciklanabilir** bir anomali suphe skoru ureten bir
karar-destek sistemidir. Hukuki hukum vermez, yatirim tavsiyesi degildir.

**Arastirma sorusu:** Cok boyutlu, robust istatistik ve makine ogrenmesi tabanli
sinyalleri, dogrulanabilir kaynaklardan gelen haber sinyalleriyle birlestirerek
yanlis pozitif oranini dusurebilir miyiz ve her skoru bilesenlerine ayirabilir miyiz?

## 2. Metodolojik Cerceve (Katmanli)

### Cekirdek
- **Robust normalizasyon (MAD).** Standart z-skoru kalin kuyruklu getiri
  dagilimlarinda uclardan asiri etkilenir. Medyan ve MAD tabanli olcekleyici
  kullanilir [Rousseeuw & Hubert 2011].
  - `Z_robust = (x - median(x)) / (1.4826 * MAD(x))`
- **Cok degiskenli aykirilik (Robust Mahalanobis / MCD).** Kovaryans matrisi
  Minimum Covariance Determinant ile robust tahmin edilir [Rousseeuw 1984].
  - `d^2 = (x - mu)^T Sigma^{-1} (x - mu) ~ chi^2_m`
- **Aykirilik (ML): Isolation Forest.** Ortalama yol uzunlugu kisaldikca anomali
  skoru 1'e yaklasir [Liu, Ting, Zhou 2008].
  - `s(x,n) = 2^{-E(h(x))/c(n)}`, `c(n) = 2 ln(n-1) + 0.5772 - 2(n-1)/n`
- **Endeks ayrismasi: Event Study / Market Model.** Endeksin kendi oynakligini
  ayristirmak icin market model artigi kullanilir [MacKinlay 1997].
  - `R_i,t = alpha_i + beta_i R_m,t + e_i,t`
  - `AR_i,t = R_i,t - (alpha_i + beta_i R_m,t)`, `CAR = sum AR`
- **Füzyon.** Kistli agirlikli toplam (`min_w L(w) s.t. sum w_j = 1, w_j >= 0`)
  veya Dempster-Shafer kanit kombinasyonu [Shafer 1976].
- **Aciklanabilirlik.** SHAP [Lundberg & Lee 2017] ve waterfall katki dagilimi
  [Ribeiro 2016].

### Likidite Baglami: Tahta Riski Onseli (v3)

Akademik motivasyon: manipulasyonun, likiditesi DUSUK hisselerde yogunlastigi
kanitlanmistir:

- Aggarwal, R. K., Wu, G. (2006). Stock Market Manipulation — Theory and
  Evidence. Journal of Business 79(4).
- Khwaja, A. I., Mian, A. (2005). Unchecked Intermediaries: Quantity Manipulation
  in an Emerging Stock Market. QJE 120(2).

Buna gore MARS v3, "cok hacimli bir hissede hacim sicramasi tek basina tahtacilik
kaniti degildir" ilkesini matematiksel bir modulasyonla modele tasir:

1. **Tahta onseli** `p ∈ [0.15, 1.0]`: medyan gunluk dolar hacmin log10 uzayinda
   ankrajlar arasinda dogrusal interpolasyonu (10M TL → 1.0; 10B TL → 0.15).
2. **Sinirli agirlik modulasyonu (veto degil):**
   `w_hacim_eff = w_hacim · (1 − 0.65·(1 − p))`, `w_amihud_eff = w_amihud · (1 − 0.50·(1 − p))`.
   Kalan agirlik gun-bazli normalizasyonla diger sinyallere dagitilir.
3. **Dogrulama (BIST, 16 Eki 2026):** BIMAS (likit katman, p=0.30) suphe skoru
   53.8 → 41.7'ye dustu; ayni gun sig/cok-sig tahtalar (KAREL p=0.82, SMART
   p=0.94) tam agirlik korudu. Derin tahtada bile endeks ayrisimasi, CLV ve MCD
   sinyalleri skoru yukseltebilir — sinyal bazli veto yoktur.

Yeni manipulasyon etiketleri:
- **CLV (kapanis konumu):** (2C − H − L)/(H − L); kapanisa dogru fiyat itme
  (marking the close) izi olarak literaturde gecer.
- **Cok gunlu desen:** CAR₅ (kumulatif anormal getiri) + hacim suresi
  (kisa/uzun pencere ortalamasi) + ayni-yon gun serisi. Pump schemelerin
  "birikim → yukseltme" evresini yakalamak icin.

### Denetimli ML Katmani (v2 eklenti, v3'te 8 oznitelikle yeniden egitildi)

Etiketli zemin gercegi olmadigindan **kontrollu sentetik etiket semasi** kullanildi:
`pump`, `subtle` → 1 (supheli); `news_legit` (mesru haber hareketi) ve normal
gunler → 0. Model: SimpleImputer(median) + StandardScaler + LogisticRegression
(class_weight=balanced). Egitim: 30 seed x 3 senaryo x 400 gun = 36.000 gozlem
(28.800 egitim / 7.200 holdout). **Holdout AUC = 0.9934** (v3: 8 oznitelik — CLV ve
cok gunlu desen dahil).

Ogrenilen katsayilar (standardize edilmis sinyaller uzerinde, v3):
`sig_index_divergence +1.27 > sig_isolation_forest +0.66 > sig_volume +0.50 >
sig_close_location +0.38 > sig_mahalanobis +0.28 > sig_pump_pattern +0.03 >
sig_volatility -0.35 > sig_liquidity -0.49`.
Dominant sinyalin endeks ayrisimasi (market model AR) olmasi, olay-calismasi
yaklasiminin (MacKinlay 1997) ML tarafindan da secildigini gosterir — yontem
tutarliligi kanitidir.

Onemli metodolojik not: AUC yalnizca sentetik ayrim uzerindedir; gercek piyasa
dogrulamasi degildir. Colab defteri (`notebooks/mars_ml_colab.ipynb`) kullanicinin
SPK kararlarindan derledigi `cases.csv` (ticker,tarih,etiket) ile gercek vaka
eklenmesine izin verir — etiketler kullanicinin sorumlulugundadir.

Gun-bazli agirlik normalizasyonu: bir sinyalin o gun verisi yoksa (or. ML modeli
yok) agirligi diger mevcut sinyallere yeniden dagitilir; boylece eksik sinyal
skoru sistematik dusurmez (kisitli optimizasyonun eksik-veri genellemesi).

### Ileri (opsiyonel)
- **Kuyruk esigi: EVT / Peaks-Over-Threshold (GPD)** [McNeil & Frey 2000].
- **Karşı-olgusal etki: BSTS / CausalImpact** [Brodersen et al. 2015].
- **Rejim/kirilma: Bayesian Online Change-Point Detection** [Adams & MacKay 2007].
- **Likidite vekili: Amihud illiquidity** [Amihud 2002].

## 3. Haber ve Guvenilirlik Modeli

Haber sinyali iki bilesenden olusur: (i) kaynagin **guvenilirligi** C in [0,1]
(VeriShield 6-kanal dogrulama motoru), (ii) haberin **polaritesi ve siddeti**
(p in {-1,0,+1} ve magnitude). Etkin haber degeri:

`news_value = polarity_signed * relevance * f(C)`

Bayes guncellemesi ile suphe skoru revize edilir:
`P(Anomali | Sinyal, Haber) ∝ P(Sinyal | Anomali) P(Anomali) * L(Haber)`

Burada `L(Haber)` dogrulanmis/onaylayan haberde supheyi dusurur, dusuk
guvenilirlikli soylentide notr kalir, celisen guvenilir haberde supheyi artirir.

## 4. Tasarim Kararlari

| Karar | Secim | Gerekce |
|-------|-------|---------|
| Veri kaynagi | yfinance (.IS) + yerel ornek arsiv | Hizli MVP, lisans riskini arsiv fallback ile yonetme |
| VeriShield | Gomulu kutuphane (AGPL), adaptor + fallback | Ayni yazar; jerarside kopukluga karsi degisken adaptor |
| Matematik | Katmanli: cekirdek zorunlu + ileri opsiyonel | 10 haftalik takvim kistinda risk yonetimi |
| Duygu modeli | Hibrit BERT + LLM fallback | Cevrimdisi determinizm + zor orneklerde esneklik |
| Dil | TR dokuman, EN kod | Ekip icin okunabilirlik + uluslararasi kod standarti |

## 5. Deneysel Bulgular

Tum metrikler SENTETIK senaryolardan turetilmistir (kontrollu karsilastirma);
gercek dunya dogrulanmis tespit basarisi degildir. Ayrinti: `docs/results.md`.

**Yontem karsilastirmasi (ROC-AUC, 10 seed):**

| Senaryo | Baseline (tekil hacim) | MARS (agirlikli) | MARS (D-S) |
|---------|------------------------|------------------|------------|
| pump (belirgin hacim sıçraması) | 1.000 | 1.000 | 0.986 |
| news_legit (meşru hareket) | 0.993 | 1.000 | 0.977 |
| subtle (sığ tahta, ince) | 0.875 | **0.974** | 0.910 |

Kritik bulgu: Belirgin hacim sıçramalarında tekil esik yeterlidir. Ancak ince/sığ
tahta senaryosunda tekil hacim sinyali zayiflar (AUC 0.875); cok boyutlu MARS
füzyonu (AUC 0.974) anlamli sekilde daha iyi ayristirir. Bu, MARS'in temel
motivasyonunu (tekil gosterge yanilgisi) destekler.

**Ablation (pump, ortalama suphe dususu):**
sig_volume (25.0) > sig_index_divergence (17.8) > sig_mahalanobis (15.0) >
sig_isolation_forest (8.9) > sig_volatility (5.6) > sig_liquidity (0.8).
Yani hacim, endeks ayrisimasi ve robust cok degiskenli uzaklik en yuksek katkiyi
saglar; bu, sinyal setinin tasarimini dogrular.

**Haber duzeltmesi:** Guvenilir (KAP) pozitif haber, hareketle ayni yonde oldugunda
supheyi ornekte %18 dusurdu (71.6 -> 58.4; haber faktoru 0.81).

**EVT:** Sentetik kalin-kuyruk (Student-t, df=4) serisinde enjekte edilen -%15
getiri EVT kuyruk sinyalinde 0.98 uretti (taban ortalama 0.05).

**BOCPD:** 3-4 sigma buyuklugundeki ortalama kaymalarini (200. gun) 0.94-1.00
olasilikla tespit etti; 2 sigma kaymayi (beklenildigi gibi) tespit etmedi.

## 6. Acik Sorular

1. Kalin kuyrukta EVT esigi ile MAD z-skoru esigi arasindaki yanlis pozitif farki?
2. D-S füzyonunda yuksek celiski (K -> 1) durumu MARS'ta nasil ele alinmali?
3. Haber guvenilirlik carpaninin kalibrasyonu nasil denetlenmeli (ground truth yok)?
4. BIST seans yapisi (tek fiyat/kesintili) gunluk OHLCV ozniteliklerini nasil etkiler?

## 7. Olasi Makale Katkilari

- BIST baglaminda aciklanabilir, robust cok boyutlu anomali skorlama cercevesi.
- Haber guvenilirlik skorunun suphe skoruna Bayesci entegrasyonu.
- Sentetik olarak dogrulanmis (etiket yoklugunda) degerlendirme protokolu.

## 8. Kaynaklar

- [1] Chandola, V., Banerjee, A., Kumar, V., 2009. Anomaly detection: A survey. ACM CSUR 41(3).
- [2] Liu, F. T., Ting, K. M., Zhou, Z. H., 2008. Isolation forest. IEEE ICDM.
- [3] Rousseeuw, P. J., Hubert, M., 2011. Robust statistics for outlier detection. WIREs DMKD 1(1).
- [4] Amihud, Y., 2002. Illiquidity and stock returns. Journal of Financial Markets 5(1).
- [5] Ribeiro, M. T., Singh, S., Guestrin, C., 2016. "Why should I trust you?". ACM SIGKDD.
- [6] MacKinlay, A. C., 1997. Event Studies in Economics and Finance. Journal of Economic Literature 35(1).
- [7] Rousseeuw, P. J., 1984. Least Median of Squares Regression. JASA 79(388).
- [8] Brodersen, K. H., et al., 2015. Inferring causal impact using Bayesian structural time-series models. Annals of Applied Statistics 9(1).
- [9] Adams, R. P., MacKay, D. J. C., 2007. Bayesian Online Changepoint Detection. arXiv:0710.3742.
- [10] McNeil, A. J., Frey, R., 2000. Estimation of tail-related risk measures for heteroscedastic financial time series. Journal of Empirical Finance 7(3).
- [11] Shafer, G., 1976. A Mathematical Theory of Evidence. Princeton University Press.
- [12] Lundberg, S. M., Lee, S.-I., 2017. A Unified Approach to Interpreting Model Predictions. NeurIPS.
- [13] SPK, 2024. Piyasa Dolandiriciligi ve Piyasa Bozucu Eylemler Rehberi.
