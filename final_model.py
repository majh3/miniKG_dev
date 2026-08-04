

from __future__ import annotations

import math
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn

try:
    from .drum import SupplyGatedDRUM
    from .facts import fact_count
    from .gate_grad_normalization import register_state as register_gate_grad_normalization_state
except ImportError:                           
    from drum import SupplyGatedDRUM
    from facts import fact_count
    from gate_grad_normalization import register_state as register_gate_grad_normalization_state


def build_model(
    facts: np.ndarray,
    entity_count: int,
    relation_count: int,
    cfg: SimpleNamespace,
    device: torch.device,
    target_relations: list[int] | None = None,
) -> SupplyGatedDRUM:
    model = SupplyGatedDRUM(
        relation_channels=2 * relation_count + 1,
        step=cfg.K,
        rules=cfg.L,
        entity_count=entity_count,
        fact_count=fact_count(facts),
        emb_size=48 if cfg.dataset == "family" else 64,
        tau_1=1.0,
        dropout=0.0,
        top_k_entities=512,
        top_k_mask=20_000,
        use_topk=True,
        supply_temperature=0.2,
        hard_threshold=0.3 if cfg.dataset == "family" else 0.5,
        rule_logit_init_scale=0.01,
        rule_parametrization="shared",
        norm_epsilon=0.0,
        norm_availability="mass",
        same_relation_channel_penalty=0.0,
        model_base="drum",
        target_table_relations=None,
    ).to(device)
    init_logit = math.log(0.8 / (1.0 - 0.8)) * 0.2
    model.gate_scope = "train_targets" if target_relations is not None else "all"
    model.gate_base_logit = float(init_logit)
    if target_relations is None:
        model.weight_param = nn.Parameter(
            torch.full((fact_count(facts), 1), init_logit, dtype=torch.float32, device=device)
        )
    else:
        global_idx = np.flatnonzero(
            np.isin(np.asarray(facts)[:, 1], np.asarray(target_relations, dtype=np.int64))
        )
        if not global_idx.size:
            raise ValueError("profiling selected no trainable facts")
        model.gate_global_index = torch.as_tensor(global_idx, dtype=torch.long, device=device)
        model.weight_param = nn.Parameter(
            torch.full((int(global_idx.size), 1), init_logit, dtype=torch.float32, device=device)
        )
    register_gate_grad_normalization_state(model)
    return model


def build_optimizers(model: SupplyGatedDRUM, cfg: SimpleNamespace) -> list[torch.optim.Optimizer]:
    rule_params = [
        param
        for name, param in model.named_parameters()
        if name != "weight_param" and param.requires_grad
    ]
    optimizers: list[torch.optim.Optimizer] = []
    if rule_params:
        optimizers.append(torch.optim.Adam(rule_params, lr=0.01 if cfg.dataset == "family" else 0.015))
                                                                                   
    optimizers.append(torch.optim.SGD(
        [model.weight_param], lr=(0.01 if cfg.dataset == "family" else 0.015) * 20.0
    ))
    return optimizers
