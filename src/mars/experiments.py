# SPDX-License-Identifier: AGPL-3.0-only
"""Deneysel degerlendirme — baseline vs MARS, ablation, stres testi.

UYARI: Gercek zemin gercegi (ground truth) bulunmadigindan metrikler SENTETIK
senaryolardan turetilir. Bu metrikler "gercek dunyada dogrulanmis tespit basarisi"
OLARAK yorumlanmamalidir; yalnizca yontemlerin sentetik anomalileri ne kadar iyi
ayirdigini gosterir (kontrollu karsilastirma).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import Settings, load_settings
from .fusion.weighted import fuse
from .fusion.evidence import fuse_evidence
from .ingestion import align_stock_market
from .models.market_model import rolling_abnormal_returns
from .signals import SIGNAL_COLUMNS, compute_signals
from .synthetic import SyntheticConfig, generate_market, generate_stock


@dataclass
class ScenarioResult:
    suspicion: pd.Series
    baseline: pd.Series
    ds: pd.Series
    labels: np.ndarray
    index: pd.Index


def run_scenario(
    seed: int,
    *,
    n_days: int = 400,
    n_injections: int = 8,
    kind: str = "pump",
    settings: Settings | None = None,
) -> ScenarioResult:
    settings = settings or load_settings()
    cfg = SyntheticConfig(n_days=n_days, seed=seed)
    stock = generate_stock(cfg, beta=1.0, n_injections=n_injections, injection_kind=kind)
    market = generate_market(cfg)
    aligned = align_stock_market(stock, market)
    sf = compute_signals(aligned, settings)
    ar = rolling_abnormal_returns(sf.frame["log_ret"], sf.frame["market_log_ret"],
                                 window=settings.estimation_window)
    f = fuse(sf.frame, abnormal_returns=ar, settings=settings)
    ds = fuse_evidence(sf.frame)

    inj = stock.attrs.get("injection_idx", np.array([], dtype=int))
    labels = np.zeros(len(sf.frame), dtype=int)
    valid_inj = [i for i in inj if i < len(labels)]
    labels[valid_inj] = 1

    # Tekil-esik baseline: hacim robust z > 3 tek basina anomali.
    baseline = sf.frame["sig_volume"].fillna(0.0)

    return ScenarioResult(
        suspicion=f.frame["suspicion"],
        baseline=baseline,
        ds=ds.reindex(f.frame.index).fillna(0.0),
        labels=labels,
        index=sf.frame.index,
    )


def _safe_auc(scores: pd.Series, labels: np.ndarray) -> float:
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, scores.to_numpy(dtype=float)))


@dataclass
class MethodComparison:
    auc: dict[str, float] = field(default_factory=dict)
    injections_mean: dict[str, float] = field(default_factory=dict)
    baseline_mean: dict[str, float] = field(default_factory=dict)


def compare_methods(
    seeds: list[int] | None = None,
    *,
    kind: str = "pump",
    settings: Settings | None = None,
) -> MethodComparison:
    seeds = seeds or list(range(1, 11))
    settings = settings or load_settings()
    acc = {"baseline": [], "mars_weighted": [], "mars_ds": []}
    inj = {"baseline": [], "mars_weighted": [], "mars_ds": []}
    base = {"baseline": [], "mars_weighted": [], "mars_ds": []}

    for s in seeds:
        r = run_scenario(s, kind=kind, settings=settings)
        acc["baseline"].append(_safe_auc(r.baseline, r.labels))
        acc["mars_weighted"].append(_safe_auc(r.suspicion, r.labels))
        acc["mars_ds"].append(_safe_auc(r.ds, r.labels))
        mask = r.labels == 1
        for k, series in (("baseline", r.baseline), ("mars_weighted", r.suspicion), ("mars_ds", r.ds)):
            inj[k].append(float(series[mask].mean()))
            base[k].append(float(series[~mask].mean()))

    return MethodComparison(
        auc={k: float(np.nanmean(v)) for k, v in acc.items()},
        injections_mean={k: float(np.mean(v)) for k, v in inj.items()},
        baseline_mean={k: float(np.mean(v)) for k, v in base.items()},
    )


def ablation(
    seeds: list[int] | None = None,
    *,
    kind: str = "pump",
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Her sinyali tek tek cikarip enjeksiyon gunlerindeki ortalama suphe dususunu olcer."""
    seeds = seeds or list(range(1, 11))
    settings = settings or load_settings()

    full_scores = []
    ablated: dict[str, list[float]] = {c: [] for c in SIGNAL_COLUMNS}

    for s in seeds:
        cfg = SyntheticConfig(n_days=400, seed=s)
        stock = generate_stock(cfg, n_injections=8, injection_kind=kind)
        market = generate_market(cfg)
        sf = compute_signals(align_stock_market(stock, market), settings)
        ar = rolling_abnormal_returns(sf.frame["log_ret"], sf.frame["market_log_ret"],
                                     window=settings.estimation_window)
        f = fuse(sf.frame, abnormal_returns=ar, settings=settings)
        inj = [i for i in stock.attrs.get("injection_idx", []) if i < len(f.frame)]
        full_scores.append(float(f.frame["suspicion"].iloc[inj].mean()))

        for col in SIGNAL_COLUMNS:
            masked = sf.frame.copy()
            masked[col] = np.nan  # sinyal cikarildi
            f_ab = fuse(masked, abnormal_returns=ar, settings=settings)
            ablated[col].append(float(f_ab.frame["suspicion"].iloc[inj].mean()))

    full_mean = float(np.mean(full_scores))
    rows = []
    for col in SIGNAL_COLUMNS:
        ab_mean = float(np.mean(ablated[col]))
        rows.append({
            "signal": col,
            "full_mean": round(full_mean, 2),
            "ablated_mean": round(ab_mean, 2),
            "importance": round(full_mean - ab_mean, 2),
        })
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def stress_test(settings: Settings | None = None) -> pd.DataFrame:
    """Farkli senaryolarda (pump vs news_legit) yontem ayrimini karsilastirir."""
    settings = settings or load_settings()
    rows = []
    for kind in ["pump", "news_legit", "subtle"]:
        cmp = compare_methods(seeds=list(range(1, 6)), kind=kind, settings=settings)
        for method in cmp.auc:
            rows.append({
                "kind": kind,
                "method": method,
                "auc": round(cmp.auc[method], 3),
                "injections_mean": round(cmp.injections_mean[method], 1),
                "baseline_mean": round(cmp.baseline_mean[method], 1),
            })
    return pd.DataFrame(rows)
