#!/usr/bin/env python3
"""Command-line entry for the self-contained final TNB path."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import torch

try:
    from .config import parse_config
    from .data import load_dataset, orientation_inverse, orientation_list
    from .train import train_once
except ImportError:  # direct script execution
    from config import parse_config
    from data import load_dataset, orientation_inverse, orientation_list
    from train import train_once


def run(cfg) -> int:
    facts, entity_count, relation_count = load_dataset(cfg)
    selected_inverse = orientation_inverse(facts, relation_count, entity_count, "forward")
    curve_path = Path(f"runs/{cfg.dataset}/curve.jsonl")
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve_path.write_text("", encoding="utf-8")

    model, args, final_metrics = train_once(
        facts, selected_inverse, cfg, 3200 if cfg.dataset == "family" else 16000,
        "final", collect_curve=True,
    )

    curve_rows = [json.loads(line) for line in curve_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = {
        "config": vars(cfg),
        "selected_inverse_relations": orientation_list(selected_inverse),
        "curve": curve_rows,
        "final": asdict(final_metrics),
        "decode_compare": [asdict(item) for item in getattr(args, "_last_decode_metrics", [final_metrics])],
    }
    path = Path(f"runs/{cfg.dataset}/output.json")
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    path = Path(f"runs/{cfg.dataset}/model_final.pt")
    torch.save(model.state_dict(), path)
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(
            {
                "selected_inverse_relations": orientation_list(selected_inverse),
                "args": {"rules": cfg.L, "step": cfg.K, "backend": "kernel_sp"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_config(argv))


if __name__ == "__main__":
    raise SystemExit(main())
