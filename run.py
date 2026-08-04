                      


from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import torch

try:
    from .config import parse_config
    from .data import load_dataset, orientation_inverse, orientation_list
    from .relation_profile import select_relations, subset_state
    from .train import train_once
except ImportError:                           
    from config import parse_config
    from data import load_dataset, orientation_inverse, orientation_list
    from relation_profile import select_relations, subset_state
    from train import train_once


def _large_graph_train_relations(dataset: str) -> list[int] | None:
    if dataset == "wikidata5m":
        return [0, 15, 3, 10, 4, 5, 6, 11, 8, 16, 13, 24, 20, 14, 38, 66, 12]
    if dataset == "freebase":
        return [2, 151, 0, 3, 4, 145, 1, 1025, 6, 96, 25, 222, 35, 29, 9, 8, 5, 646, 32, 315, 12, 11, 1028, 648, 989, 106, 371, 10, 317, 7, 316, 31, 344, 3395, 225, 3713, 3711, 3715, 30, 730, 1168, 60, 172, 97, 61, 447, 74, 311, 75, 647, 22, 152, 236, 1165, 1162, 1164, 3712, 649, 11009, 34, 1163, 987, 533, 154, 271, 23, 21, 153, 20, 3026, 985, 47, 140, 107, 224, 645, 46, 36, 6100, 59, 62, 7042, 986, 231, 84, 86, 3191, 2877, 1448, 3520, 3517]
    return None


def run(cfg) -> int:
    facts, entity_count, relation_count = load_dataset(cfg)
    selected_inverse = orientation_inverse(facts, relation_count, entity_count, "forward")
    curve_path = Path(f"runs/{cfg.dataset}/curve.jsonl")
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve_path.write_text("", encoding="utf-8")

    large_graph = cfg.dataset in {"freebase", "wikidata5m"}
    model, args, final_metrics = train_once(
        facts, selected_inverse, cfg,
        3200 if cfg.dataset == "family" else 16000 if cfg.dataset == "yago3-10" else 750 if cfg.dataset == "wikidata5m" else 900,
        "profile" if large_graph else "final",
        collect_curve=not large_graph,
        target_relations=_large_graph_train_relations(cfg.dataset),
    )
    profile = None
    if large_graph:
        selected = select_relations(final_metrics, cfg.profile_threshold)
        profile = {
            "threshold": cfg.profile_threshold,
            "selected_relations": selected,
            "base_fact_rate": final_metrics.fact_rate,
        }
        if selected:
            state = subset_state(model, facts, selected)
            model, args, final_metrics = train_once(
                facts, selected_inverse, cfg, 300, "profile_continue",
                target_relations=selected,
                initial_state=state,
            )

    curve_rows = [json.loads(line) for line in curve_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = {
        "config": vars(cfg),
        "selected_inverse_relations": orientation_list(selected_inverse),
        "training": getattr(args, "_training_status", {}),
        "relation_profile": profile,
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
                "training": getattr(args, "_training_status", {}),
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
