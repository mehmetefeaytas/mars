#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
"""Deneyleri yeniden uretir ve sonuclari docs/results.md'ye yazar.

Kullanim:
    PYTHONPATH=src .venv/bin/python scripts/run_experiments.py
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mars.experiments import ablation, compare_methods, stress_test  # noqa: E402


def main() -> int:
    lines = [f"# MARS Deneysel Sonuclar\n", f"_Uretim: {datetime.now():%Y-%m-%d %H:%M}_\n"]
    lines.append("> Not: Metrikler SENTETIK senaryolardan turetilmistir; gercek dunya "
                 "dogrulanmis tespit basarisi degildir.\n")

    for kind in ["pump", "news_legit", "subtle"]:
        cmp = compare_methods(kind=kind)
        lines.append(f"\n## Senaryo: `{kind}`\n")
        lines.append("| Yontem | AUC | Enjeksiyon ort. | Taban ort. |")
        lines.append("|--------|-----|-----------------|------------|")
        for m in cmp.auc:
            lines.append(f"| {m} | {cmp.auc[m]:.3f} | {cmp.injections_mean[m]:.1f} | {cmp.baseline_mean[m]:.2f} |")

    lines.append("\n## Ablation (pump) — sinyal onem sirasi\n")
    abl = ablation(kind="pump")
    lines.append(abl.to_markdown(index=False))

    lines.append("\n## Stres testi (tum senaryolar)\n")
    st = stress_test()
    lines.append(st.to_markdown(index=False))

    out = ROOT / "docs" / "results.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Yazildi: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
