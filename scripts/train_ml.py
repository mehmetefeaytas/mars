#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
"""ML siniflandiriciyi egitir ve kaydeder (data/models/mars_ml.joblib).

Kullanim:
    PYTHONPATH=src .venv/bin/python scripts/train_ml.py [--seeds 40] [--kinds pump,subtle,news_legit]

Egitim seti sentetik senaryolardan uretilir:
  pump, subtle -> etiket 1 (supheli), news_legit -> etiket 0 (mesru hareket).
Colab'da ayni egitim icin: notebooks/mars_ml_colab.ipynb
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mars import ml  # noqa: E402
from mars.config import load_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(prog="mars-train")
    parser.add_argument("--seeds", type=int, default=40, help="Sentetik senaryo seed sayisi")
    parser.add_argument("--kinds", type=str, default="pump,subtle,news_legit")
    args = parser.parse_args()

    settings = load_settings()
    seeds = list(range(1, args.seeds + 1))
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())

    print(f"Egitim seti uretiliyor: {len(seeds)} seed x {kinds} ...")
    trained = ml.train(seeds, settings=settings)
    print(f"Ornek sayisi: {trained.artifact['meta']['n_train']} egitim / "
          f"{trained.artifact['meta']['n_test']} holdout")
    print(f"Holdout AUC: {trained.auc:.4f}")
    print("Katsayilar:")
    for c, w in trained.artifact["meta"]["coefficients"].items():
        print(f"  {c:24s} {w:+.3f}")

    path = ml.save_model(trained, settings=settings)
    print(f"Model kaydedildi: {path}")
    print(f"Meta: {path.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
