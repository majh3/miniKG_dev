                      


from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import torch

try:
    from .config import parse_config
    from .data import load_dataset, orientation_inverse, orientation_list
    from .relation_profile import select_relation_head, select_relations, subset_state
    from .train import train_once
except ImportError:                           
    from config import parse_config
    from data import load_dataset, orientation_inverse, orientation_list
    from relation_profile import select_relation_head, select_relations, subset_state
    from train import train_once


def run(cfg) -> int:
    facts, entity_count, relation_count = load_dataset(cfg)
    selected_inverse = orientation_inverse(facts, relation_count, entity_count, "forward")
    curve_path = Path(f"runs/{cfg.dataset}/curve.jsonl")
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve_path.write_text("", encoding="utf-8")

    large_graph = cfg.dataset in {"freebase", "wikidata5m"}
    train_relations = (
        select_relation_head(facts, 0.7 if cfg.dataset == "wikidata5m" else 0.8)
        if large_graph else None
    )
    model, args, final_metrics = train_once(
        facts, selected_inverse, cfg,
        cfg.steps,
        "profile" if large_graph else "final",
        collect_curve=not large_graph,
        target_relations=train_relations,
    )
    profile = None
    if large_graph:
        selected = select_relations(final_metrics, 0.8)
        profile = {
            "threshold": 0.8,
            "trained_relations": train_relations,
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
