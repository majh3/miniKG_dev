

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
    if set(values) != {"kappa", "K", "L", "profile_threshold", "decode_prune_threshold"}:
        parser.error(
            "the recipe must contain exactly kappa, K, L, profile_threshold, and decode_prune_threshold"
        )
    if any(float(values[key]) <= 0 for key in ("kappa", "K", "L", "profile_threshold", "decode_prune_threshold")):
        parser.error("all five hyperparameters must be positive")
    if any(float(values[key]) > 1 for key in ("profile_threshold", "decode_prune_threshold")):
        parser.error("profile and decode pruning thresholds must not exceed one")
    return SimpleNamespace(
        dataset=ns.dataset,
        kappa=float(values["kappa"]),
        K=int(values["K"]),
        L=int(values["L"]),
        profile_threshold=float(values["profile_threshold"]),
        decode_prune_threshold=float(values["decode_prune_threshold"]),
    )
