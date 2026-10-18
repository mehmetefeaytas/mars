# SPDX-License-Identifier: AGPL-3.0-only
"""Denetimli ML katmani — ogrenilmis anomali siniflandirici.

Yaklasim:
  * Etiketli zemin gercegi olmadigindan egitim seti KONTROLLU sentetik senaryolardan
    uretilir: "pump" ve "subtle" (sug tahta) gunleri = 1 (supheli), "news_legit"
    (mesru haber hareketi) ve normal gunler = 0.
  * Model: SimpleImputer + StandardScaler + LogisticRegression (class_weight=balanced).
    Olasilik cikisi `sig_ml` olarak füzyona girer; interpretability icin
    katsayilari aciklanabilir kalir.
  * Colab'da ayni egitim `notebooks/mars_ml_colab.ipynb` ile tekrar uretilebilir;
    kullanici etiketli CSV (ticker,tarih,etiket) ekleyerek gercek vaka ile
    zenginlestirebilir (or. SPK kararlarindan derlenen gunler).

Not: Model dosyasi yoksa sistem denetimsiz cekirdek ile calismaya devam eder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Settings, load_settings
from .ingestion import align_stock_market
from .models.market_model import rolling_abnormal_returns
from .synthetic import SyntheticConfig, generate_market, generate_stock

# Etiket semasi: senaryo turu -> etiket (1 = supheli, 0 = mesru/normal)
SCENARIO_LABELS = {"pump": 1, "subtle": 1, "news_legit": 0}


@dataclass
class TrainedModel:
    artifact: dict          # {model, columns, meta}
    columns: list[str]
    auc: float


def default_model_path(settings: Settings | None = None) -> Path:
    settings = settings or load_settings()
    return settings.data_dir / "models" / "mars_ml.joblib"


def build_dataset(
    seeds: list[int] | None = None,
    *,
    kinds: tuple[str, ...] = ("pump", "subtle", "news_legit"),
    n_days: int = 400,
    n_injections: int = 8,
    settings: Settings | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sentetik senaryolardan (X, y) egitim tablosu uretir."""
    settings = settings or load_settings()
    seeds = seeds or list(range(1, 41))
    rows: list[dict] = []
    labels: list[int] = []

    # Tembel import: signals <-> ml dongusel bagimliligini kirar.
    from .signals import compute_signals  # noqa: PLC0415

    for seed in seeds:
        for kind in kinds:
            cfg = SyntheticConfig(n_days=n_days, seed=seed)
            stock = generate_stock(cfg, beta=1.0, n_injections=n_injections, injection_kind=kind)
            market = generate_market(cfg)
            sf = compute_signals(align_stock_market(stock, market), settings)
            inj = set(int(i) for i in stock.attrs.get("injection_idx", []) if i < len(sf.frame))
            base_cols = [c for c in sf.frame.columns if c.startswith("sig_") and c != "sig_ml"]
            for i, (date, row) in enumerate(sf.frame.iterrows()):
                rows.append({c: row.get(c) for c in base_cols})
                if i in inj:
                    labels.append(SCENARIO_LABELS.get(kind, 0))
                else:
                    labels.append(0)

    return pd.DataFrame(rows), np.asarray(labels, dtype=int)


def train(
    seeds: list[int] | None = None,
    *,
    settings: Settings | None = None,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> TrainedModel:
    """Modeli egitir; holdout AUC ile birlikte dondurur (kaydetmez)."""
    settings = settings or load_settings()
    X, y = build_dataset(seeds, settings=settings)

    split = int(len(X) * (1.0 - test_fraction))
    # Zaman sirasi bozulmasin: ilk kisim egitim, son kisim holdout.
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y[:split], y[split:]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=random_state)),
    ])
    model.fit(X_tr, y_tr)

    if len(np.unique(y_te)) > 1:
        proba = model.predict_proba(X_te)[:, 1]
        auc = float(roc_auc_score(y_te, proba))
    else:
        auc = float("nan")

    columns = list(X.columns)
    artifact = {
        "model": model,
        "columns": columns,
        "meta": {
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "holdout_auc": auc,
            "coefficients": {
                c: float(w) for c, w in zip(columns, model.named_steps["clf"].coef_[0])
            },
            "intercept": float(model.named_steps["clf"].intercept_[0]),
            "label_schema": SCENARIO_LABELS,
        },
    }
    return TrainedModel(artifact=artifact, columns=columns, auc=auc)


def save_model(trained: TrainedModel, path: Path | None = None, settings: Settings | None = None) -> Path:
    settings = settings or load_settings()
    path = path or default_model_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained.artifact, path)
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(trained.artifact["meta"], indent=2, ensure_ascii=False),
                         encoding="utf-8")
    return path


def load_model(settings: Settings | None = None) -> TrainedModel | None:
    """Kayitli modeli yukler; yoksa None (sistem denetimsiz calisir)."""
    settings = settings or load_settings()
    path = default_model_path(settings)
    if not path.exists():
        return None
    try:
        artifact = joblib.load(path)
        return TrainedModel(
            artifact=artifact,
            columns=list(artifact.get("columns", [])),
            auc=float(artifact.get("meta", {}).get("holdout_auc", float("nan"))),
        )
    except Exception:  # noqa: BLE001
        return None


def predict_scores(
    signal_frame: pd.DataFrame, trained: TrainedModel | None, settings: Settings | None = None
) -> pd.Series:
    """`sig_ml` olasilik serisi (0..1). Model yoksa tumu NaN."""
    out = pd.Series(np.nan, index=signal_frame.index, name="sig_ml")
    if trained is None:
        return out
    cols = [c for c in trained.columns if c in signal_frame.columns]
    if not cols:
        return out
    X = signal_frame[cols]
    mask = X.notna().any(axis=1)
    if not mask.any():
        return out
    try:
        proba = trained.artifact["model"].predict_proba(X[mask])[:, 1]
        out.loc[mask] = proba
    except Exception:  # noqa: BLE001
        pass
    return out
