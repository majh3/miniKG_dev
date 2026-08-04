

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def parse_config(argv: list[str] | None = None) -> SimpleNamespace:
    parser = argparse.ArgumentParser(description="Reproduce MiniKG champion compression")
    parser.add_argument(
        "--dataset", required=True,
        choices=["family", "yago3-10", "wikidata5m", "freebase"],
    )
    ns = parser.parse_args(argv)
    values = json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))
    if set(values) != {"kappa", "K", "L", "steps"}:
        parser.error("the recipe must contain exactly kappa, K, L, and steps")
    if set(values["steps"]) != {"family", "yago3-10", "wikidata5m", "freebase"}:
        parser.error("steps must cover all four datasets")
    if any(float(values[key]) <= 0 for key in ("kappa", "K", "L")) or any(
        int(step) <= 0 for step in values["steps"].values()
    ):
        parser.error("kappa, K, L, and steps must be positive")
    return SimpleNamespace(
        dataset=ns.dataset,
        steps=int(values["steps"][ns.dataset]),
        kappa=float(values["kappa"]),
        K=int(values["K"]),
        L=int(values["L"]),
    )
