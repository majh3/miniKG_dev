"""Local DRUM model used by src_final.

This is a focused copy of the active DRUM path from src_unify.fullgraph_unify.
The core equations are intentionally unchanged; src_final only owns the smaller
training/decode surface around it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def normalize_dense_last_dim(state: torch.Tensor, eps: float, availability: str) -> torch.Tensor:
    denom = state.sum(dim=-1, keepdim=True)
    if eps > 0:
        normalized = state / denom.clamp_min(eps)
    else:
        denom_safe = torch.where(denom > 0, denom, torch.ones_like(denom))
        normalized = state / denom_safe
        normalized = torch.where(denom > 0, normalized, torch.zeros_like(normalized))
    if availability == "mass":
        return normalized * denom.clamp_max(1.0)
    if availability == "none":
        return normalized
    raise ValueError(f"unknown proof_norm_availability: {availability}")


def normalize_sparse3d(state: torch.Tensor, eps: float, availability: str) -> torch.Tensor:
    state = state.coalesce()
    idx = state.indices()
    val = state.values()
    keep_input = val != 0
    idx = idx[:, keep_input]
    val = val[keep_input]
    batch_size, rule_count, _ = state.shape
    denom = torch.zeros(batch_size, rule_count, device=val.device, dtype=val.dtype)
    denom.index_put_((idx[0], idx[1]), val, accumulate=True)
    if eps > 0:
        denom = denom.clamp_min(eps)
    val_norm = val / denom[idx[0], idx[1]]
    if availability == "mass":
        val_norm = val_norm * denom.clamp_max(1.0)[idx[0], idx[1]]
    elif availability != "none":
        raise ValueError(f"unknown proof_norm_availability: {availability}")
    keep = val_norm != 0
    idx = idx[:, keep]
    val_norm = val_norm[keep]
    return torch.sparse_coo_tensor(idx, val_norm, state.shape, device=val.device).coalesce()


def prune_sparse3d_active_entities(state: torch.Tensor, topk_nodes: int) -> torch.Tensor:
    if topk_nodes <= 0:
        return state
    state = state.coalesce()
    idx = state.indices()
    val = state.values()
    if val.numel() <= topk_nodes:
        return state
    entity_count = state.shape[2]
    # Family/WN18 etc. may set kernel_topk_nodes larger than |E|; clamp k.
    k = min(int(topk_nodes), int(entity_count))
    if k <= 0:
        return state
    entity_score = torch.zeros(entity_count, dtype=val.dtype, device=val.device)
    entity_score.index_add_(0, idx[2], val.detach().abs())
    # Do not count active entities with a graph-sized boolean temporary here.
    # ``topk`` already preserves every nonzero entity when their count is at
    # most ``topk_nodes``; the existing ``keep.all`` fast path below then
    # returns the original state.  On FB this avoids a transient mask spanning
    # all ~305M entities (hundreds of MiB) at the peak of each proof graph.
    _, top_idx = torch.topk(entity_score, k=k)
    keep_entity = torch.zeros(entity_count, dtype=torch.bool, device=val.device)
    keep_entity[top_idx] = True
    keep = keep_entity[idx[2]]
    if bool(keep.all().item()):
        return state
    return torch.sparse_coo_tensor(idx[:, keep], val[keep], state.shape, device=val.device).coalesce()


def broadcast_index(index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    if dim < 0:
        dim = src.dim() + dim
    if index.dim() == 1:
        for _ in range(dim):
            index = index.unsqueeze(0)
    for _ in range(index.dim(), src.dim()):
        index = index.unsqueeze(-1)
    return index.expand(src.size())


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1, dim_size: int | None = None) -> torch.Tensor:
    index = broadcast_index(index, src, dim)
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)


class MaskedSigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, temperature, mask_indices):
        out = torch.sigmoid(logits / temperature)
        if mask_indices is not None and mask_indices.numel() > 0:
            out[mask_indices.long(), 0] = 0.0
        ctx.save_for_backward(out)
        ctx.temperature = temperature
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        grad_logits = grad_output * out * (1.0 - out) / ctx.temperature
        return grad_logits, None, None


def masked_sigmoid(logits: torch.Tensor, temperature: float, mask_indices: torch.Tensor | None) -> torch.Tensor:
    return MaskedSigmoidFunction.apply(logits, temperature, mask_indices)


def block_is_in(values: torch.Tensor, targets: torch.Tensor, block_size: int = 10_000_000) -> torch.Tensor:
    parts = []
    for start in range(0, values.numel(), block_size):
        block = values[start : start + block_size]
        hits = torch.nonzero(torch.isin(block, targets), as_tuple=False).flatten()
        parts.append(hits + start)
    if not parts:
        return torch.empty((0,), dtype=torch.long, device=values.device)
    return torch.cat(parts)


def sparse_step_from_onehot(A, graph_pack, entity_count, relation_count, tau_1, edge_weight, wot_i=False):
    head, tail, rel, mask, rule_logits = graph_pack
    active_entities = torch.unique(torch.nonzero(A, as_tuple=False)[:, 1])

    edge_idx = block_is_in(head, active_entities)
    h = torch.index_select(head, 0, edge_idx)
    t = torch.index_select(tail, 0, edge_idx)
    r = torch.index_select(rel, 0, edge_idx)
    m = torch.index_select(mask, 0, edge_idx)
    ew = torch.index_select(edge_weight, 0, edge_idx.to(edge_weight.device)).to(A.device)

    inv_idx = block_is_in(tail, active_entities)
    inv_h = torch.index_select(tail, 0, inv_idx)
    inv_t = torch.index_select(head, 0, inv_idx)
    inv_r = torch.index_select(rel, 0, inv_idx)
    inv_m = torch.index_select(mask, 0, inv_idx)
    inv_ew = torch.index_select(edge_weight, 0, inv_idx.to(edge_weight.device)).to(A.device)

    rule = torch.softmax(rule_logits / tau_1, dim=-1)

    a_h = torch.index_select(A, 1, h)
    rule_h = torch.index_select(rule[:, :, :relation_count], 2, r)
    values = torch.einsum("bm,blm->blm", a_h, rule_h) * m.unsqueeze(0).unsqueeze(0)
    values = values * ew.t().unsqueeze(0)
    out_forward = scatter_sum(values, t.long(), dim=2, dim_size=entity_count)

    a_t = torch.index_select(A, 1, inv_h)
    rule_t = torch.index_select(rule[:, :, relation_count : 2 * relation_count], 2, inv_r)
    inv_values = torch.einsum("bm,blm->blm", a_t, rule_t) * inv_m.unsqueeze(0).unsqueeze(0)
    inv_values = inv_values * inv_ew.t().unsqueeze(0)
    out_backward = scatter_sum(inv_values, inv_t.long(), dim=2, dim_size=entity_count)

    out_identity = None
    if not wot_i:
        out_identity = torch.einsum("bm,bl->blm", A, rule[:, :, -1])
    return out_identity, out_forward, out_backward


def sparse_step_from_state(A, graph_pack, entity_count, relation_count, tau_1, edge_weight, top_k, top_k_mask, use_topk, wot_i=False):
    head, tail, rel, mask, rule_logits = graph_pack
    active_entities = torch.unique(torch.nonzero(A.sum(1), as_tuple=False)[:, 1])
    if use_topk:
        k = min(top_k, A.shape[-1])
        top_entities = torch.unique(torch.topk(A.sum(1), k=k).indices.reshape(-1))
        if top_entities.shape[0] < active_entities.shape[0]:
            active_entities = top_entities

    edge_idx = block_is_in(head, active_entities)
    h = torch.index_select(head, 0, edge_idx)
    t = torch.index_select(tail, 0, edge_idx)
    r = torch.index_select(rel, 0, edge_idx)
    m = torch.index_select(mask, 0, edge_idx)
    ew = torch.index_select(edge_weight, 0, edge_idx.to(edge_weight.device)).to(A.device).squeeze(-1)

    inv_idx = block_is_in(tail, active_entities)
    inv_h = torch.index_select(tail, 0, inv_idx)
    inv_t = torch.index_select(head, 0, inv_idx)
    inv_r = torch.index_select(rel, 0, inv_idx)
    inv_m = torch.index_select(mask, 0, inv_idx)
    inv_ew = torch.index_select(edge_weight, 0, inv_idx.to(edge_weight.device)).to(A.device).squeeze(-1)

    rule = torch.softmax(rule_logits / tau_1, dim=-1)

    a_h = torch.index_select(A, -1, h)
    if use_topk:
        k = min(top_k_mask, a_h.shape[-1])
        a_top, a_pos = torch.topk(a_h, k=k)
        rule_h = torch.index_select(rule[:, :, :relation_count], 2, r)
        rule_top = torch.gather(rule_h, -1, a_pos)
        values = a_top * rule_top * m[a_pos]
        values = values * ew[a_pos]
        out_forward = scatter_sum(values, t[a_pos].long(), dim=2, dim_size=entity_count)
    else:
        rule_h = torch.index_select(rule[:, :, :relation_count], 2, r)
        values = a_h * rule_h * m
        values = values * ew
        out_forward = scatter_sum(values, t.long(), dim=2, dim_size=entity_count)

    a_t = torch.index_select(A, -1, inv_h)
    if use_topk:
        k = min(top_k_mask, a_t.shape[-1])
        a_top, a_pos = torch.topk(a_t, k=k)
        rule_t = torch.index_select(rule[:, :, relation_count : 2 * relation_count], 2, inv_r)
        rule_top = torch.gather(rule_t, -1, a_pos)
        inv_values = a_top * rule_top * inv_m[a_pos]
        inv_values = inv_values * inv_ew[a_pos]
        out_backward = scatter_sum(inv_values, inv_t[a_pos].long(), dim=2, dim_size=entity_count)
    else:
        rule_t = torch.index_select(rule[:, :, relation_count : 2 * relation_count], 2, inv_r)
        inv_values = a_t * rule_t * inv_m
        inv_values = inv_values * inv_ew
        out_backward = scatter_sum(inv_values, inv_t.long(), dim=2, dim_size=entity_count)

    out_identity = None
    if not wot_i:
        out_identity = torch.einsum("ble,bl->ble", A, rule[:, :, -1])
    return out_identity, out_forward, out_backward


def tclm_invert_rule_logits(rule_logits: torch.Tensor, relation_count: int) -> torch.Tensor:
    fwd = rule_logits[..., :relation_count]
    inv = rule_logits[..., relation_count : 2 * relation_count]
    ident = rule_logits[..., 2 * relation_count :]
    swapped = torch.cat([inv, fwd, ident], dim=-1)
    return torch.flip(swapped, dims=[1])


class SupplyGatedDRUM(nn.Module):
    def __init__(
        self,
        relation_channels,
        step,
        rules,
        entity_count,
        fact_count,
        emb_size,
        tau_1,
        dropout,
        top_k_entities,
        top_k_mask,
        use_topk,
        supply_temperature,
        hard_threshold,
        rule_logit_init_scale,
        rule_parametrization,
        norm_epsilon,
        norm_availability,
        same_relation_channel_penalty,
        model_base="drum",
        target_table_relations=None,
    ):
        super().__init__()
        self.relation_channels = relation_channels
        self.relation_count = (relation_channels - 1) // 2
        self.step = step
        self.rules = rules
        self.entity_count = entity_count
        self.fact_count = fact_count
        self.tau_1 = tau_1
        self.top_k_entities = top_k_entities
        self.top_k_mask = top_k_mask
        self.use_topk = use_topk
        self.supply_temperature = supply_temperature
        self.hard_threshold = hard_threshold
        self.model_base = model_base
        self.rule_parametrization = rule_parametrization
        self.norm_epsilon = norm_epsilon
        self.norm_availability = norm_availability
        self.same_relation_channel_penalty = same_relation_channel_penalty
        self.edge_weight_mask = None
        self.weight_transform = None
        self.weight_override = None
        self.legacy_bool_edge_mask = False
        self.legacy_hard_supply_logits = False
        # Gate scope: "all" trains a logit per fact; "train_targets" trains logits only
        # for target-relation facts (non-target gates frozen at gate_base_logit). Set up
        # by final_model.build_model; keeps huge-graph gate params/optimizer states small.
        self.gate_scope = "all"
        self.gate_global_index = None  # sorted global fact indices of trainable gates
        self.gate_base_logit = 0.0
        # Training enables this only when the gate optimizer accepts sparse COO
        # gradients (chunked sparse NBE on huge graphs). Forward values unchanged.
        self.sparse_gate_parameter_grad = False

        self.emb = nn.Parameter(torch.Tensor(relation_channels, emb_size))
        nn.init.kaiming_uniform_(self.emb, a=np.sqrt(5))
        self.lstm = nn.ModuleList(nn.LSTM(emb_size, emb_size, 1, bidirectional=True) for _ in range(rules))
        self.linear = nn.Linear(2 * emb_size, relation_channels)
        with torch.no_grad():
            self.linear.weight.mul_(rule_logit_init_scale)
            self.linear.bias.mul_(rule_logit_init_scale)
        self.dropout = nn.Dropout(dropout)
        self.weight_param = nn.Parameter(torch.ones(fact_count, 1) * 2.5)
        if self.rule_parametrization == "target_table":
            table_rels = target_table_relations
            if table_rels is not None:
                table_rels = sorted({int(r) for r in table_rels})
                if not table_rels:
                    table_rels = None
            if table_rels is None:
                rows = self.relation_count
                index_map = torch.arange(self.relation_count, dtype=torch.long)
            else:
                rows = len(table_rels)
                index_map = torch.full((self.relation_count,), -1, dtype=torch.long)
                index_map[torch.tensor(table_rels, dtype=torch.long)] = torch.arange(rows, dtype=torch.long)
            self.register_buffer("target_table_index", index_map)
            self.target_rule_logits = nn.Parameter(torch.empty(rows, step, rules, relation_channels))
            nn.init.normal_(self.target_rule_logits, mean=0.0, std=rule_logit_init_scale)
        else:
            self.target_table_index = None
            self.target_rule_logits = None

    def full_gate_logits(self):
        if self.gate_scope == "all":
            return self.weight_param
        full = torch.full(
            (self.fact_count, 1),
            float(self.gate_base_logit),
            dtype=self.weight_param.dtype,
            device=self.weight_param.device,
        )
        return torch.index_put(full, (self.gate_global_index,), self.weight_param)

    def gate_logits(self, global_idx):
        global_idx = global_idx.long().view(-1)
        if self.gate_scope == "all":
            if bool(getattr(self, "sparse_gate_parameter_grad", False)):
                return F.embedding(global_idx, self.weight_param, sparse=True).view(-1)
            return self.weight_param.view(-1)[global_idx]
        pos = torch.searchsorted(self.gate_global_index, global_idx)
        pos = pos.clamp(max=int(self.gate_global_index.numel()) - 1)
        hit = self.gate_global_index[pos] == global_idx
        out = torch.full(
            global_idx.shape,
            float(self.gate_base_logit),
            dtype=self.weight_param.dtype,
            device=self.weight_param.device,
        )
        if bool(hit.any()):
            if bool(getattr(self, "sparse_gate_parameter_grad", False)):
                selected = F.embedding(pos[hit], self.weight_param, sparse=True).view(-1)
            else:
                selected = self.weight_param.view(-1)[pos[hit]]
            out = out.masked_scatter(hit, selected)
        return out

    @property
    def weight(self):
        if self.weight_override is not None:
            weight = self.weight_override
            if self.edge_weight_mask is not None:
                weight = weight.clone()
                if self.edge_weight_mask.dtype == torch.bool:
                    mask = self.edge_weight_mask
                    if weight.dim() == 1 and mask.dim() == 2:
                        mask = mask.view(-1)
                    weight = torch.where(mask, torch.zeros_like(weight), weight)
                else:
                    mask = self.edge_weight_mask.detach().long().view(-1)
                    if weight.dim() == 1:
                        weight[mask] = 0.0
                    else:
                        weight[mask, 0] = 0.0
            if weight.dtype == torch.bool:
                weight = weight.to(dtype=self.weight_param.dtype)
            if self.weight_transform is not None:
                mode, threshold, temperature = self.weight_transform
                if mode == "decision":
                    weight = torch.sigmoid((weight - threshold) / temperature)
                elif mode == "straight_through":
                    hard_weight = (weight >= threshold).to(weight.dtype)
                    weight = hard_weight.detach() + weight - weight.detach()
            return weight
        if self.weight_override is None and self.gate_scope == "train_targets" and self.weight_transform is None:
            # Compute sigmoid only for the [M] trainable gates, then embed into a detached
            # base buffer (no grad flows through non-target positions → no [N] backward alloc).
            base_val = float(torch.sigmoid(torch.tensor(self.gate_base_logit / self.supply_temperature)))
            weight = torch.full(
                (self.fact_count, 1), base_val,
                dtype=self.weight_param.dtype, device=self.weight_param.device,
            )
            target_weight = torch.sigmoid(self.weight_param / self.supply_temperature)
            weight = torch.index_put(weight, (self.gate_global_index,), target_weight)
            if self.edge_weight_mask is not None:
                if self.edge_weight_mask.dtype == torch.bool:
                    weight = torch.where(self.edge_weight_mask, torch.zeros_like(weight), weight)
                else:
                    mask = self.edge_weight_mask.detach().long().view(-1)
                    weight = weight.clone()
                    weight[mask, 0] = 0.0
            return weight
        logits = self.full_gate_logits()
        if self.edge_weight_mask is not None:
            if self.edge_weight_mask.dtype == torch.bool:
                weight = torch.sigmoid(logits / self.supply_temperature)
                weight = torch.where(self.edge_weight_mask, torch.zeros_like(weight), weight)
            else:
                weight = masked_sigmoid(logits, self.supply_temperature, self.edge_weight_mask)
        else:
            weight = torch.sigmoid(logits / self.supply_temperature)
        if self.weight_transform is not None:
            mode, threshold, temperature = self.weight_transform
            if mode == "decision":
                weight = torch.sigmoid((weight - threshold) / temperature)
            elif mode == "straight_through":
                hard_weight = (weight >= threshold).to(weight.dtype)
                weight = hard_weight.detach() + weight - weight.detach()
        return weight

    def _shared_rule_logits(self, target_rel):
        base = torch.index_select(self.emb, 0, target_rel)
        seq = torch.stack([base] * (self.step + 1), dim=1)
        seq[:, -1, :] = self.emb[-1]
        seq = seq.transpose(1, 0)
        logits = []
        for lstm in self.lstm:
            out, _ = lstm(seq)
            logits.append(self.linear(out.transpose(1, 0)[:, :-1, :]))
        return torch.stack(logits, dim=2)

    def build_rule_logits(self, target_rel):
        if self.rule_parametrization == "target_table":
            row = self.target_table_index[target_rel.long()]
            mapped = row >= 0
            if bool(mapped.all().item()):
                logits = self.target_rule_logits[row]
            else:
                # Non-target relations reach here only via negative-scoring paths;
                # they fall back to the shared parametrization while target
                # relations keep independent table rows.
                logits = self._shared_rule_logits(target_rel).clone()
                if bool(mapped.any().item()):
                    logits[mapped] = self.target_rule_logits[row[mapped]]
        else:
            logits = self._shared_rule_logits(target_rel)
        if self.same_relation_channel_penalty > 0:
            logits = logits.clone()
            rows = torch.arange(target_rel.shape[0], device=target_rel.device)
            logits[rows, :, :, target_rel.long()] -= self.same_relation_channel_penalty
            logits[rows, :, :, target_rel.long() + self.relation_count] -= self.same_relation_channel_penalty
        if getattr(self, "_tclm_invert_rules", False):
            logits = tclm_invert_rule_logits(logits, self.relation_count)
        elif getattr(self, "_tclm_selective_inverse", None) is not None:
            inv_mask = self._tclm_selective_inverse[target_rel.long()].bool()
            if inv_mask.any():
                inverted = tclm_invert_rule_logits(logits, self.relation_count)
                logits = torch.where(inv_mask.view(-1, 1, 1, 1), inverted, logits)
        return logits

    def forward(self, heads, rels, graph, is_training=False):
        state = F.one_hot(heads.long(), self.entity_count).bool()
        all_rule_logits = self.build_rule_logits(rels)
        edge_weight = self.weight
        states = []
        for hop in range(self.step):
            rule_logits = all_rule_logits[:, hop, :, :]
            graph_pack = (graph.head, graph.tail, graph.rel, graph.mask, rule_logits)
            if hop == 0:
                identity, forward, backward = sparse_step_from_onehot(
                    state,
                    graph_pack,
                    self.entity_count,
                    self.relation_count,
                    self.tau_1,
                    edge_weight,
                    wot_i=True,
                )
            else:
                identity, forward, backward = sparse_step_from_state(
                    states[-1],
                    graph_pack,
                    self.entity_count,
                    self.relation_count,
                    self.tau_1,
                    edge_weight,
                    self.top_k_entities,
                    self.top_k_mask,
                    self.use_topk,
                    wot_i=False,
                )
            if self.model_base == "mmdrum":
                parts = [forward, backward] if identity is None else [identity, forward, backward]
                state = torch.stack(parts, dim=0).max(dim=0).values
                state = normalize_dense_last_dim(state, self.norm_epsilon, self.norm_availability)
            else:
                state = forward + backward if identity is None else identity + forward + backward
                state = state.clamp_min(0).clamp_max(1)
            if is_training:
                state = self.dropout(state)
            states.append(state)
        if self.model_base == "mmdrum":
            return states[-1].max(dim=1).values
        return states[-1].sum(dim=1)
