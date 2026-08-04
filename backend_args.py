

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch


def build_backend_args(
    cfg: SimpleNamespace,
    orientation_inverse: np.ndarray,
    device: torch.device,
    decode_relations: list[int] | None = None,
) -> SimpleNamespace:
    large_graph = cfg.dataset in {"freebase", "wikidata5m"}
    args = SimpleNamespace(
        dataset=cfg.dataset,
        large_graph=large_graph,
        kernel_topk_edges=(
            50 if cfg.dataset == "freebase"
            else 100 if cfg.dataset == "wikidata5m"
            else (1 << 60) if cfg.dataset == "family" else 100
        ),
        kernel_topk_nodes=(
            50_000 if cfg.dataset == "freebase"
            else 0 if cfg.dataset in {"family", "wikidata5m"} else 100_000
        ),
        decode_query_prune="rule_first_hop" if large_graph else "none",
        decode_query_rule_threshold=(0.2 if large_graph else 0.0),
        decode_batch_size=(64 if cfg.dataset == "freebase" else 512 if large_graph else 4096 if cfg.dataset == "family" else 1024),
        decode_proof_microbatch_size=(64 if cfg.dataset == "freebase" else 256 if cfg.dataset == "wikidata5m" else 0),
        threshold_qsth_query_source=("supply_entities" if cfg.dataset == "family" else "supply_heads"),
        decode_relations=",".join(str(relation) for relation in decode_relations or []),
    )
    args._tnb_false_main_batch_count = 0
    args._tnb_proof_supply_override = None
    args._tnb_proof_graph_global_edge_indices = None
    args._query_orientation_inverse_np = orientation_inverse
    args._query_orientation_inverse_tensor = torch.zeros(
        len(orientation_inverse), dtype=torch.bool, device=device
    )
    for relation, inverse in enumerate(orientation_inverse):
        args._query_orientation_inverse_tensor[relation] = bool(inverse)
    return args
