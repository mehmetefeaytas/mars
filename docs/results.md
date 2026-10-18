# MARS Deneysel Sonuclar

_Uretim: 2026-10-18 12:10_

> Not: Metrikler SENTETIK senaryolardan turetilmistir; gercek dunya dogrulanmis tespit basarisi degildir.


## Senaryo: `pump`

| Yontem | AUC | Enjeksiyon ort. | Taban ort. |
|--------|-----|-----------------|------------|
| baseline | 1.000 | 1.0 | 0.07 |
| mars_weighted | 1.000 | 73.2 | 15.52 |
| mars_ds | 0.986 | 100.0 | 12.97 |

## Senaryo: `news_legit`

| Yontem | AUC | Enjeksiyon ort. | Taban ort. |
|--------|-----|-----------------|------------|
| baseline | 0.993 | 0.8 | 0.07 |
| mars_weighted | 1.000 | 67.7 | 15.47 |
| mars_ds | 0.977 | 100.0 | 12.53 |

## Senaryo: `subtle`

| Yontem | AUC | Enjeksiyon ort. | Taban ort. |
|--------|-----|-----------------|------------|
| baseline | 0.875 | 0.4 | 0.07 |
| mars_weighted | 0.974 | 48.3 | 15.45 |
| mars_ds | 0.910 | 84.5 | 15.52 |

## Ablation (pump) — sinyal onem sirasi

| signal               |   full_mean |   ablated_mean |   importance |
|:---------------------|------------:|---------------:|-------------:|
| sig_volume           |       73.17 |          48.18 |        24.99 |
| sig_index_divergence |       73.17 |          55.33 |        17.84 |
| sig_mahalanobis      |       73.17 |          58.17 |        15    |
| sig_isolation_forest |       73.17 |          64.26 |         8.91 |
| sig_volatility       |       73.17 |          67.54 |         5.64 |
| sig_liquidity        |       73.17 |          72.39 |         0.79 |

## Stres testi (tum senaryolar)

| kind       | method        |   auc |   injections_mean |   baseline_mean |
|:-----------|:--------------|------:|------------------:|----------------:|
| pump       | baseline      | 1     |               1   |             0.1 |
| pump       | mars_weighted | 1     |              72.4 |            15.1 |
| pump       | mars_ds       | 0.976 |             100   |            14.7 |
| news_legit | baseline      | 0.996 |               0.8 |             0.1 |
| news_legit | mars_weighted | 1     |              67.8 |            15.2 |
| news_legit | mars_ds       | 0.961 |             100   |            15.2 |
| subtle     | baseline      | 0.887 |               0.4 |             0.1 |
| subtle     | mars_weighted | 0.968 |              49.3 |            15.2 |
| subtle     | mars_ds       | 0.877 |              77.8 |            14.3 |
