

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch


def build_backend_args(
    cfg: SimpleNamespace,
    orientation_inverse: np.ndarray,
    device: torch.device,
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
        decode_query_rule_threshold=(cfg.decode_prune_threshold if large_graph else 0.0),
        decode_batch_size=(64 if cfg.dataset == "freebase" else 512 if large_graph else 4096 if cfg.dataset == "family" else 1024),
        decode_proof_microbatch_size=(64 if cfg.dataset == "freebase" else 256 if cfg.dataset == "wikidata5m" else 0),
        threshold_qsth_query_source=("supply_entities" if cfg.dataset == "family" else "supply_heads"),
        decode_relations=(
            "0,3,5,6,10,12,15,16,20,24,38" if cfg.dataset == "wikidata5m"
            else "2,151,0,3,4,145,1,1025,6,96,25,222,35,29,9,8,5,646,32,315,12,11,1028,648,989,106,371,10,317,7,316,31,344,3395,225,3713,3711,3715,30,730,1168,60,172,97,61,447,74,311,75,647,22,152,236,1165,1162,1164,3712,649,11009,34,1163,987,533,154,271,23,21,153,20,3026,985,47,140,107,224,645,46,36,6100,59,62,7042,986,231,84,86,3191,2877,1448,3520,3517"
            if cfg.dataset == "freebase" else ""
        ),
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
