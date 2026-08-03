"""Load the only three experimental hyperparameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def parse_config(argv: list[str] | None = None) -> SimpleNamespace:
    parser = argparse.ArgumentParser(description="Reproduce MiniKG Family/YAGO3-10 champion compression")
    parser.add_argument("--dataset", required=True, choices=["family", "yago3-10"])
    ns = parser.parse_args(argv)
    values = json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))
    if set(values) != {"kappa", "K", "L"}:
        parser.error("the recipe must contain exactly kappa, K, and L")
    if float(values["kappa"]) <= 0 or int(values["K"]) <= 0 or int(values["L"]) <= 0:
        parser.error("kappa, K, and L must be positive")
    return SimpleNamespace(
        dataset=ns.dataset,
        kappa=float(values["kappa"]),
        K=int(values["K"]),
        L=int(values["L"]),
    )
