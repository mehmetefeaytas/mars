# SPDX-License-Identifier: AGPL-3.0-only
"""Komut satiri arayuzu."""

from __future__ import annotations

import argparse
import json

from .config import load_settings
from .pipeline import analyze


def pipeline_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mars-pipeline", description="MARS anomali analizi")
    parser.add_argument("ticker", help="BIST ticker (or. THYAO.IS)")
    parser.add_argument("--no-network", action="store_true", help="Ag erisimini kapat")
    parser.add_argument("--json", action="store_true", help="JSON cikti")
    args = parser.parse_args(argv)

    settings = load_settings()
    result = analyze(args.ticker, settings=settings, allow_network=not args.no_network)
    latest = result.latest()
    payload = {
        "ticker": result.ticker,
        "data_source": result.data_source,
        "date": str(result.fusion.frame.index[-1].date()),
        "suspicion": round(float(latest["suspicion"]), 1),
        "data_confidence": round(float(latest["data_confidence"]), 1),
        "waterfall": result.waterfall(),
        "summary": result.summary(),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload["summary"])
        contribs = ", ".join(f"{w['signal']}=+{w['points']}" for w in payload["waterfall"])
        print(f"Katkilar: {contribs}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pipeline_cli())
