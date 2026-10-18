# MARS — Calisma Logu

> Bu dosya **append-only**'dir. Yapilan her islem tarih/saat ile eklenir.
> Format: `| Tarih | Saat | Yapilan Islem | Ilgili Dosya/Artifact | Sonuc |`

| Tarih | Saat | Yapilan Islem | Ilgili Dosya/Artifact | Sonuc |
|-------|------|---------------|-----------------------|-------|
| 2026-10-18 | 11:36 | Proje raporu ve on sunum PDF'leri okundu; VeriShield reposu incelendi | `MARS_Proje_Raporu.pdf`, `MARS_BLMS405.pdf`, verishield-core README | Mimari, 6 asamali pipeline ve 3 ek gereksinim (haber skoru, akademik matematik, LLM polarite) cikarildi |
| 2026-10-18 | 11:54 | Akademik yontem dogrulama arastirmasi yapildi | Event Study (MacKinlay 1997), BSTS/CausalImpact (Brodersen 2015), Dempster-Shafer, HF Turkce duygu modelleri | Katmanli matematik yigini onaylandi |
| 2026-10-18 | 12:05 | Karar toplantisi: veri kaynagi, VeriShield entegrasyonu, matematik kapsami, duygu modeli, dil | — | yfinance+arsiv; gomulu kutuphane (AGPL); katmanli cekirdek+ileri; hibrit BERT+LLM; TR dokuman + EN kod |
| 2026-10-18 | 12:10 | Faz 0 basladi: repo iskelesi, pyproject/requirements, .gitignore, .env.example | `pyproject.toml`, `requirements.txt`, `.gitignore`, `.env.example` | Ortam tanimlari hazir |
| 2026-10-18 | 12:12 | log.md ve academic.md olusturuldu | `log.md`, `academic.md` | Surdurulebilir kayit altyapisi kuruldu |
| 2026-10-18 | 12:30 | Faz 0 tamamlandi; verishield-core submodule eklendi; embedded skorlama test edildi | `vendor/verishield-core`, `pyproject.toml` | Commit + push yapildi (master) |
| 2026-10-18 | 12:40 | Ortam kuruldu (Python 3.12 venv, uv) | `.venv` | Cekirdek bagimliliklar yuklendi |
| 2026-10-18 | 12:55 | Faz 1: ingestion yazildi (yfinance .IS, cache, takvim hizalama) + sentetik uretici | `src/mars/ingestion.py`, `src/mars/synthetic.py` | THYAO.IS gercek verisiyle dogrulandi (249 gun) |
| 2026-10-18 | 13:05 | Faz 2: robust ozellikler (log getiri, MAD z, aralik, Amihud) | `src/mars/features.py` | Hazir |
| 2026-10-18 | 13:20 | Faz 3: anomali cekirdek (MCD-Mahalanobis, Isolation Forest, market-model CAR) + sinyal montaji | `src/mars/anomaly/`, `src/mars/models/market_model.py`, `src/mars/signals.py` | Sentetik enjeksiyonlar net ayristi (hacim 1.0, endeks 0.98) |
| 2026-10-18 | 13:45 | Faz 4: haber modulu (kaynak semasi, hibrit BERT/lexicon duygu, LLM kopru, VeriShield adaptor) | `src/mars/news/` | Embedded VeriShield skoru alindi; lexicon duygu dogrulandi |
| 2026-10-18 | 14:05 | Faz 5: füzyon (kisitli agirlikli + Dempster-Shafer), kalibrasyon, aciklanabilirlik + pipeline/CLI | `src/mars/fusion/`, `src/mars/explain.py`, `src/mars/pipeline.py`, `src/mars/cli.py` | Enjeksiyon suphesi 71-81 vs taban 16; DS enjeksiyon 100; haber duzeltmesi 71.6->58.4 |
| 2026-10-18 | 14:30 | Faz 7: ileri yontemler (EVT POT/GPD, BOCPD) | `src/mars/anomaly/evt.py`, `src/mars/anomaly/changepoint.py` | EVT enjeksiyonda 0.98; BOCPD 4-5 sigma kaymayi 0.94-1.0 tespit |
| 2026-10-18 | 14:50 | Faz 6: deney modulu (baseline vs MARS, ablation, stres testi) | `src/mars/experiments.py` | Ince senaryoda MARS 0.974 > tekil 0.875 (AUC) |
| 2026-10-18 | 15:05 | Streamlit arayuzu | `app.py` | Gercek+sentetik mod, risk karti, selale grafigi |
| 2026-10-18 | 15:20 | Test paketi yazildi ve duzeltildi (34 test) | `tests/` | Tum testler gecti (`pytest -q`) |
| 2026-10-18 | 15:30 | Deney scripti + sonuc raporu, README, academic bulgular | `scripts/run_experiments.py`, `docs/results.md`, `README.md` | Fazlar 6-8 commit + push |
