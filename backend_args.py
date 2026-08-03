"""Create mutable runtime state; scientific settings live at their equations."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch


def build_backend_args(
    cfg: SimpleNamespace,
    orientation_inverse: np.ndarray,
    device: torch.device,
) -> SimpleNamespace:
    args = SimpleNamespace(dataset=cfg.dataset)
    args._tnb_false_main_batch_count = 0
    args._tnb_proof_supply_override = None
    args._tnb_proof_graph_global_edge_indices = None
    args._query_orientation_inverse_np = orientation_inverse
    args._query_orientation_inverse_tensor = torch.tensor(
        orientation_inverse,
        dtype=torch.bool,
        device=device,
    )
    return args
