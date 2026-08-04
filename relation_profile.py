

from __future__ import annotations

import numpy as np
import torch

try:
    from .metrics import DecodeMetrics
except ImportError:
    from metrics import DecodeMetrics


def select_relations(metrics: DecodeMetrics, threshold: float) -> list[int]:
    return [
        int(row["relation"])
        for row in metrics.relation_rows
        if (int(row["supply"]) + int(row["missing"]) + int(row["extra"]))
        / max(int(row["facts"]), 1) < float(threshold)
    ]


def subset_state(model, facts: np.ndarray, relations: list[int]) -> dict[str, torch.Tensor]:
    pass                                                                      
    selected_global = np.flatnonzero(
        np.isin(np.asarray(facts)[:, 1], np.asarray(relations, dtype=np.int64))
    )
    selected = torch.as_tensor(selected_global, dtype=torch.long, device=model.weight_param.device)
    if model.gate_scope == "train_targets":
        positions = torch.searchsorted(model.gate_global_index, selected)
        if positions.numel() and (
            int(positions.max()) >= int(model.gate_global_index.numel())
            or not torch.equal(model.gate_global_index[positions], selected)
        ):
            raise RuntimeError("profiled relations are outside the trained relation set")
        gates = model.weight_param.detach().index_select(0, positions).cpu()
    else:
        gates = model.weight_param.detach().index_select(0, selected).cpu()
    return {
        key: (gates if key == "weight_param" else value.detach().cpu())
        for key, value in model.state_dict().items()
    }
