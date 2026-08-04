

                                                                                
                                            
   

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    from .drum import (
        SupplyGatedDRUM,
        normalize_sparse3d,
        prune_sparse3d_active_entities,
    )
    from .paths import SRC_DIR
except ImportError:                           
    from drum import SupplyGatedDRUM, normalize_sparse3d, prune_sparse3d_active_entities
    try:
        from paths import SRC_DIR
    except ImportError:
        SRC_DIR = Path(__file__).resolve().parent / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

_KERNEL_OPS = None


def kernel_ops():
    global _KERNEL_OPS
    if _KERNEL_OPS is None:
        from model_mmDRUM_kernel_sp import (
            FastLogFunctionSparse3DTopK,
            build_csr_structure,
            max_sp_3d,
            sparse3d_max,
            sum_sp_3d,
            sum_sparse3d_concat,
        )

        _KERNEL_OPS = (
            FastLogFunctionSparse3DTopK,
            build_csr_structure,
            max_sp_3d,
            sparse3d_max,
            sum_sp_3d,
            sum_sparse3d_concat,
        )
    return _KERNEL_OPS


def use_logit_edge_weight_path(model: SupplyGatedDRUM, args) -> bool:
    pass                                                                    

                                                                          
                                                                             
                                                                          
                                                                              
                                                                            
                                                             
       
    sparse_edge_grad = os.environ.get("FASTLOG_SPARSE_EDGE_GRAD", "0") == "1"
    return (
        sparse_edge_grad
        and model.weight_override is None
        and model.weight_transform is None
        and str(getattr(model, "gate_scope", "all")) in ("all", "train_targets")
    )


def validate_sparse_score_path(edge_scores: torch.Tensor) -> None:
    pass                                                                         
    if (
        os.environ.get("FASTLOG_SPARSE_EDGE_GRAD", "0") == "1"
        and torch.is_grad_enabled()
        and edge_scores.requires_grad
    ):
        raise RuntimeError(
            "FASTLOG_SPARSE_EDGE_GRAD=1 requires the logit edge-weight path for "
            "differentiable supplies; weight_override, weight_transform, or a "
            "legacy boolean edge mask made that path unavailable"
        )


class TrainTargetLogitEdgeScores(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_logits: torch.Tensor, gate_global_index: torch.Tensor, fact_count: int, base_logit: float):
        gate_global_index = gate_global_index.detach().long().view(-1)
        ctx.save_for_backward(gate_global_index)
        ctx.local_shape = tuple(local_logits.shape)
        full = local_logits.new_full((int(fact_count),), float(base_logit))
        if gate_global_index.numel() > 0:
            full.index_copy_(0, gate_global_index, local_logits.detach().reshape(-1))
        return full

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (gate_global_index,) = ctx.saved_tensors
        local_count = int(gate_global_index.numel())
        if grad_output is None:
            grad_local = torch.zeros(local_count, dtype=torch.float32, device=gate_global_index.device)
        elif grad_output.is_sparse:
            grad = grad_output.coalesce()
            idx = grad.indices()
            if idx.shape[0] != 1:
                raise RuntimeError(f"expected 1-D sparse edge-score grad, got sparse shape {tuple(grad.shape)}")
            edge_idx = idx[0].long()
            pos = torch.searchsorted(gate_global_index, edge_idx)
            in_bounds = pos < local_count
            safe_pos = pos.clamp(max=max(local_count - 1, 0))
            hit = in_bounds & (gate_global_index[safe_pos] == edge_idx)
            grad_local = grad.values().new_zeros(local_count)
            if bool(hit.any().item()):
                grad_local.index_add_(0, safe_pos[hit], grad.values()[hit])
        else:
            grad_local = grad_output.reshape(-1).index_select(0, gate_global_index)
        return grad_local.reshape(ctx.local_shape), None, None, None


class SparseTrainTargetLogitEdgeScores(torch.autograd.Function):
    pass                                                               

                                                                                  
                                                                           
                                                                            
                                                             
       

    @staticmethod
    def forward(ctx, local_logits: torch.Tensor, gate_global_index: torch.Tensor, fact_count: int, base_logit: float):
        gate_global_index = gate_global_index.detach().long().view(-1)
        ctx.save_for_backward(gate_global_index)
        ctx.local_shape = tuple(local_logits.shape)
        full = local_logits.new_full((int(fact_count),), float(base_logit))
        if gate_global_index.numel() > 0:
            full.index_copy_(0, gate_global_index, local_logits.detach().reshape(-1))
        return full

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (gate_global_index,) = ctx.saved_tensors
        local_count = int(gate_global_index.numel())
        if grad_output is None:
            return None, None, None, None
        if not grad_output.is_sparse:
            raise RuntimeError("sparse train-target gate bridge received a dense global gradient")
        grad = grad_output.coalesce()
        idx = grad.indices()
        if idx.shape[0] != 1:
            raise RuntimeError(f"expected 1-D sparse edge-score grad, got sparse shape {tuple(grad.shape)}")
        edge_idx = idx[0].long()
        pos = torch.searchsorted(gate_global_index, edge_idx)
        in_bounds = pos < local_count
        safe_pos = pos.clamp(max=max(local_count - 1, 0))
        hit = in_bounds & (gate_global_index[safe_pos] == edge_idx)
        local_pos = safe_pos[hit]
        local_values = grad.values()[hit].reshape(-1, 1)
        grad_local = torch.sparse_coo_tensor(
            local_pos.view(1, -1),
            local_values,
            ctx.local_shape,
            dtype=local_values.dtype,
            device=local_values.device,
        ).coalesce()
        return grad_local, None, None, None


def cached_train_target_edge_scores(model: SupplyGatedDRUM) -> torch.Tensor:
    pass                                                            

                                                                               
                                                                             
                                                                             
                                                                               
                                                                             
                                                                             
                                                                             
       
    param = model.weight_param
    key = (
        id(param),
        int(param._version),
        int(model.gate_global_index.data_ptr()),
        int(model.fact_count),
        float(model.gate_base_logit),
        bool(getattr(model, "sparse_gate_parameter_grad", False)),
    )
    cached = getattr(model, "_train_target_edge_scores_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]

                                                                              
    model._train_target_edge_scores_cache = None
    bridge = (
        SparseTrainTargetLogitEdgeScores
        if bool(getattr(model, "sparse_gate_parameter_grad", False))
        else TrainTargetLogitEdgeScores
    )
    scores = bridge.apply(
        param,
        model.gate_global_index,
        int(model.fact_count),
        float(model.gate_base_logit),
    )
    model._train_target_edge_scores_cache = (key, scores)
    return scores


def clear_cached_train_target_edge_scores(model: SupplyGatedDRUM) -> None:
    pass                                                              

                                                                               
                                                                           
                                                                             
       
    model._train_target_edge_scores_cache = None


@contextmanager
def hard_supply_context(model: SupplyGatedDRUM, hard_supply: torch.Tensor):
    if getattr(model, "legacy_hard_supply_logits", False):
        old_param = model.weight_param
        hard_logits = torch.full_like(old_param, -20.0)
        hard_logits[hard_supply.view(-1, 1)] = 20.0
        model.weight_param = nn.Parameter(hard_logits, requires_grad=False)
        try:
            yield
        finally:
            model.weight_param = old_param
        return

    old_override = model.weight_override
    n = int(hard_supply.numel())
    dev = model.weight_param.device
    dtype = model.weight_param.dtype
                                                                            
                                                                            
                                                                              
    parked_param = None
    parked_index = None
    huge = n >= 50_000_000 and dev.type == "cuda"

    if hard_supply.dtype == torch.bool:
        if huge:
            print(
                json.dumps(
                    {
                        "event": "hard_supply_override_build_start",
                        "facts": n,
                        "note": "CPU float build; park weight_param on CPU",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
                                                                     
            parked_param = model.weight_param
            model.weight_param = nn.Parameter(parked_param.detach().to("cpu"), requires_grad=False)
            if getattr(model, "gate_global_index", None) is not None:
                parked_index = model.gate_global_index
                model.gate_global_index = parked_index.detach().to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                                                                  
            flat_bool = hard_supply.detach().view(-1).to("cpu")
            hard_cpu = torch.empty((n,), dtype=dtype, device="cpu")
            for start in range(0, n, 2_000_000):
                end = min(start + 2_000_000, n)
                hard_cpu[start:end] = flat_bool[start:end].to(dtype=dtype)
            del flat_bool
            hard_weight = hard_cpu.view(-1, 1).to(device=dev, non_blocking=False)
            del hard_cpu
            print(
                json.dumps(
                    {"event": "hard_supply_override_ready", "facts": n, "device": str(dev)},
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            hard_weight = torch.empty((n, 1), dtype=dtype, device=dev)
            flat_bool = hard_supply.view(-1)
            if flat_bool.device != hard_weight.device:
                for start in range(0, n, 2_000_000):
                    end = min(start + 2_000_000, n)
                    hard_weight[start:end, 0] = flat_bool[start:end].to(
                        device=dev, dtype=dtype
                    )
            else:
                for start in range(0, n, 2_000_000):
                    end = min(start + 2_000_000, n)
                    hard_weight[start:end, 0] = flat_bool[start:end].to(dtype=dtype)
            del flat_bool
    else:
        hard_weight = hard_supply.to(device=dev, dtype=dtype).view(-1, 1)

    model.weight_override = hard_weight
    try:
        yield
    finally:
        model.weight_override = old_override
        del hard_weight
        if parked_param is not None:
            model.weight_param = parked_param
        if parked_index is not None:
            model.gate_global_index = parked_index
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@contextmanager
def masked_edge_context(model: SupplyGatedDRUM, edge_indices: torch.Tensor):
    old_mask = model.edge_weight_mask
    edge_indices = edge_indices.detach().long().view(-1)
    if edge_indices.numel() == 0:
        yield
        return
    if getattr(model, "legacy_bool_edge_mask", False):
        mask = torch.zeros_like(model.weight_param, dtype=torch.bool)
        mask[edge_indices, 0] = True
        if old_mask is not None:
            if old_mask.dtype == torch.bool:
                mask = mask | old_mask
            else:
                mask[old_mask.long(), 0] = True
        model.edge_weight_mask = mask
        try:
            yield
        finally:
            model.edge_weight_mask = old_mask
        return

    if old_mask is not None:
        if old_mask.dtype == torch.bool:
            mask = old_mask.clone()
            mask[edge_indices, 0] = True
        else:
            mask = torch.unique(torch.cat([old_mask, edge_indices], dim=0))
    else:
        mask = torch.unique(edge_indices)
    model.edge_weight_mask = mask
    try:
        yield
    finally:
        model.edge_weight_mask = old_mask


@contextmanager
def proof_graph_masked_edge_context(
    model: SupplyGatedDRUM,
    edge_indices: torch.Tensor,
    args,
):
    pass                                                                      
    graph_global_indices = getattr(
        args,
        "_tnb_proof_graph_global_edge_indices",
        None,
    )
    if graph_global_indices is None:
        with masked_edge_context(model, edge_indices):
            yield
        return

    graph_global_indices = graph_global_indices.detach().long().view(-1)
    requested = edge_indices.detach().long().view(-1)
    if graph_global_indices.numel() == 0 or requested.numel() == 0:
        mapped = requested.new_empty((0,))
    else:
        positions = torch.searchsorted(graph_global_indices, requested)
        in_bounds = positions < int(graph_global_indices.numel())
        safe_positions = positions.clamp(max=int(graph_global_indices.numel()) - 1)
        hits = in_bounds & (graph_global_indices[safe_positions] == requested)
        mapped = positions[hits]
    with masked_edge_context(model, mapped):
        yield


@contextmanager
def proof_supply_context(model: SupplyGatedDRUM, args):
    old_transform = model.weight_transform
    old_override = model.weight_override
    proof_override = getattr(args, "_tnb_proof_supply_override", None)
    if proof_override is not None:
        model.weight_override = proof_override
    try:
        yield
    finally:
        model.weight_transform = old_transform
        model.weight_override = old_override


def concat_score_batches(parts: list[torch.Tensor]) -> torch.Tensor:
    if not parts:
        raise ValueError("concat_score_batches requires at least one part")
    if not parts[0].is_sparse:
        return torch.cat(parts, dim=0)
    indices = []
    values = []
    row_offset = 0
    entity_count = int(parts[0].shape[1])
    dtype = parts[0].dtype
    device = parts[0].device
    for part in parts:
        part = part.coalesce()
        if part._nnz() > 0:
            idx = part.indices().clone()
            idx[0] += row_offset
            indices.append(idx)
            values.append(part.values())
        row_offset += int(part.shape[0])
    if not indices:
        empty_idx = torch.empty((2, 0), dtype=torch.long, device=device)
        empty_val = torch.empty((0,), dtype=dtype, device=device)
        return torch.sparse_coo_tensor(empty_idx, empty_val, [row_offset, entity_count], device=device).coalesce()
    return torch.sparse_coo_tensor(
        torch.cat(indices, dim=1),
        torch.cat(values, dim=0),
        [row_offset, entity_count],
        device=device,
    ).coalesce()


def compact_kernel_graph_for_decode(model: SupplyGatedDRUM, graph) -> int:
    pass                                                                      
    if getattr(model, "_kernel_csr", None) is None:
        raise RuntimeError("decode graph compaction requires a cached kernel CSR")
    if getattr(model, "edge_weight_mask", None) is not None:
        raise RuntimeError("decode graph compaction refuses a dynamic edge mask")
    if bool(getattr(graph, "_decode_csr_only", False)):
        return 0

    tensors = [
        getattr(graph, "head", None),
        getattr(graph, "rel", None),
        getattr(graph, "tail", None),
        getattr(graph, "mask", None),
    ]
    storages: dict[int, int] = {}
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        storage = tensor.untyped_storage()
        storages.setdefault(int(storage.data_ptr()), int(storage.nbytes()))

    graph.head = None
    graph.rel = None
    graph.tail = None
    graph.mask = None
    graph.e2triple = None
    graph.triple2e = None
    graph.r2triple = None
    graph._decode_csr_only = True
    return int(sum(storages.values()))


def kernel_sparse_scores(model: SupplyGatedDRUM, heads: torch.Tensor, rels: torch.Tensor, graph, args, is_training: bool) -> torch.Tensor:
    (
        FastLogFunctionSparse3DTopK,
        build_csr_structure,
        max_sp_3d,
        sparse3d_max,
        sum_sp_3d,
        sum_sparse3d_concat,
    ) = kernel_ops()
    decode_csr_only = bool(getattr(graph, "_decode_csr_only", False))
    if decode_csr_only and getattr(model, "edge_weight_mask", None) is not None:
        raise RuntimeError("compacted decode graph cannot apply a dynamic edge mask")

    gate_scope = str(getattr(model, "gate_scope", "all"))
    sparse_edge_grad = os.environ.get("FASTLOG_SPARSE_EDGE_GRAD", "0") == "1"
    use_logit_edge_weight = use_logit_edge_weight_path(model, args)
    edge_mask_indices = None
    edge_mask_restore = None
    if use_logit_edge_weight and model.edge_weight_mask is not None:
        if model.edge_weight_mask.dtype == torch.bool:
            use_logit_edge_weight = False
        else:
            edge_mask_indices = torch.unique(model.edge_weight_mask.detach().long().view(-1))
            if edge_mask_indices.numel() > 0:
                edge_mask_restore = graph.mask[edge_mask_indices].clone()
                graph.mask[edge_mask_indices] = False
                if getattr(graph, "mask_float", None) is not None:
                    graph.mask_float[edge_mask_indices] = 0.0

    if not decode_csr_only:
        e2triple, triple2e, r2triple = graph.e2triple, graph.triple2e, graph.r2triple
    try:
        if decode_csr_only:
            if getattr(model, "_kernel_csr", None) is None:
                raise RuntimeError("compacted decode graph lost its cached kernel CSR")
        else:
            cache_key = (e2triple[0].data_ptr(), triple2e[1].data_ptr(), r2triple[0].data_ptr())
            if getattr(model, "_kernel_csr_key", None) != cache_key:
                model._kernel_csr = build_csr_structure(e2triple[0], triple2e[1], r2triple[0], model.entity_count)
                model._kernel_csr_key = cache_key

        batch_size = heads.shape[0]
        rule_count = model.rules
        entity_count = model.entity_count
        state = torch.sparse_coo_tensor(
            torch.stack(
                [
                    torch.arange(batch_size, device=heads.device, dtype=torch.long).repeat_interleave(rule_count),
                    torch.arange(rule_count, device=heads.device, dtype=torch.long).repeat(batch_size),
                    heads.long().repeat_interleave(rule_count),
                ],
                dim=0,
            ),
            torch.ones(batch_size * rule_count, device=heads.device, dtype=torch.float32),
            [batch_size, rule_count, entity_count],
            is_coalesced=True,
        )
        rule_logits = model.build_rule_logits(rels)
        if use_logit_edge_weight:
            edge_weight_scale = 1.0 / max(float(model.supply_temperature), 1e-12)
            if gate_scope == "train_targets":
                if sparse_edge_grad:
                    edge_scores = cached_train_target_edge_scores(model)
                else:
                    edge_scores = model.full_gate_logits().reshape(-1)
            elif sparse_edge_grad:
                edge_scores = model.weight_param
            else:
                edge_scores = model.weight_param.view(-1)
            edge_weight_is_score = False
        else:
            edge_scores = model.weight
            validate_sparse_score_path(edge_scores)
            if is_training and torch.is_grad_enabled() and not sparse_edge_grad:
                edge_scores = edge_scores.reshape(-1).clone()
            else:
                edge_scores = edge_scores.reshape(-1)
            edge_weight_is_score = True
            edge_weight_scale = 1.0
        mask_values = graph.mask_float if getattr(graph, "mask_float", None) is not None else None
        if mask_values is None:
                                                                                     
                                                                                    
            mask_values = torch.ones((), dtype=torch.float32, device=heads.device)
        (
            ori_row_ptr,
            ori_col_ind,
            ori_r_ind,
            order_ori,
            inv_row_ptr,
            inv_col_ind,
            inv_r_ind,
            order_inv,
            ori_row_group_ptr,
            ori_group_rel,
            ori_group_edge_start,
            ori_group_edge_count,
            inv_row_group_ptr,
            inv_group_rel,
            inv_group_edge_start,
            inv_group_edge_count,
        ) = model._kernel_csr
        for hop in range(model.step):
            state = prune_sparse3d_active_entities(state, int(args.kernel_topk_nodes))
            w_t = torch.softmax(rule_logits[:, hop, :, :] / model.tau_1, dim=-1)
            out_ori, out_inv, out_ind = FastLogFunctionSparse3DTopK.apply(
                state.indices()[0],
                state.indices()[1],
                state.indices()[2],
                state.values(),
                w_t,
                ori_row_ptr,
                ori_col_ind,
                ori_r_ind,
                order_ori,
                inv_row_ptr,
                inv_col_ind,
                inv_r_ind,
                order_inv,
                ori_row_group_ptr,
                ori_group_rel,
                ori_group_edge_start,
                ori_group_edge_count,
                inv_row_group_ptr,
                inv_group_rel,
                inv_group_edge_start,
                inv_group_edge_count,
                mask_values,
                edge_scores,
                batch_size,
                rule_count,
                entity_count,
                model.relation_count,
                False,
                int(args.kernel_topk_edges),
                0,
                edge_weight_is_score,
                edge_weight_scale,
            )
            state = normalize_sparse3d(sum_sparse3d_concat(out_ori, out_inv, out_ind), 0.0, "mass")
            if is_training:
                state = torch.sparse_coo_tensor(state.indices(), model.dropout(state.values()), state.shape).coalesce()
        return sum_sp_3d(state)
    finally:
        if edge_mask_restore is not None:
            graph.mask[edge_mask_indices] = edge_mask_restore
            if getattr(graph, "mask_float", None) is not None:
                graph.mask_float[edge_mask_indices] = edge_mask_restore.to(dtype=graph.mask_float.dtype)


def bounded_drum_scores(model: SupplyGatedDRUM, heads: torch.Tensor, rels: torch.Tensor, graph, args, is_training: bool) -> torch.Tensor:
    raw = kernel_sparse_scores(model, heads, rels, graph, args, is_training)
    if raw.is_sparse:
        raw = raw.to_dense()
    if getattr(model, "model_base", "drum") == "mmdrum":
        return raw.clamp_min(0.0).clamp_max(1.0)
    return 1.0 - torch.exp(-raw.clamp_min(0.0))


def bound_sparse_or_dense_scores(model: SupplyGatedDRUM, raw: torch.Tensor) -> torch.Tensor:
    if not raw.is_sparse:
        if getattr(model, "model_base", "drum") == "mmdrum":
            return raw.clamp_min(0.0).clamp_max(1.0)
        return 1.0 - torch.exp(-raw.clamp_min(0.0))
    raw = raw.coalesce()
    values = raw.values()
    if getattr(model, "model_base", "drum") == "mmdrum":
        values = values.clamp_min(0.0).clamp_max(1.0)
    else:
        values = 1.0 - torch.exp(-values.clamp_min(0.0))
    return torch.sparse_coo_tensor(raw.indices(), values, raw.shape, device=values.device).coalesce()




def bound_sparse_scores(raw: torch.Tensor) -> torch.Tensor:
    pass                                                                              
                                                                                  
    if not raw.is_sparse:
        return 1.0 - torch.exp(-raw.clamp_min(0.0))
    raw = raw.coalesce()
    values = 1.0 - torch.exp(-raw.values().clamp_min(0.0))
    return torch.sparse_coo_tensor(raw.indices(), values, raw.shape, device=values.device).coalesce()

def proof_for_targets(model: SupplyGatedDRUM, targets: torch.Tensor, graph, args, is_training: bool) -> torch.Tensor:
    return bounded_drum_scores(model, targets[:, 0], targets[:, 1], graph, args, is_training=is_training)


def proof_scores_for_targets(model: SupplyGatedDRUM, targets: torch.Tensor, graph, args, is_training: bool) -> torch.Tensor:
    raw = kernel_sparse_scores(model, targets[:, 0], targets[:, 1], graph, args, is_training)
    scores = bound_sparse_or_dense_scores(model, raw)
    if not scores.is_sparse:
        return scores[torch.arange(targets.shape[0], device=targets.device), targets[:, 2].long()]
    scores = scores.coalesce()
    idx = scores.indices()
    val = scores.values()
    entity_count = scores.shape[1]
    sparse_keys = idx[0].long() * entity_count + idx[1].long()
    if sparse_keys.numel() == 0:
        return torch.zeros(targets.shape[0], dtype=val.dtype, device=val.device)
    order = torch.argsort(sparse_keys)
    sorted_keys = sparse_keys[order]
    target_keys = torch.arange(targets.shape[0], device=targets.device, dtype=torch.long) * entity_count + targets[:, 2].long()
    pos = torch.searchsorted(sorted_keys, target_keys)
    in_bounds = pos < sorted_keys.numel()
    safe_pos = pos.clamp(max=max(sorted_keys.numel() - 1, 0))
    hit = in_bounds & (sorted_keys[safe_pos] == target_keys)
    out = torch.zeros(targets.shape[0], dtype=val.dtype, device=val.device)
    if bool(hit.any().item()):
        out[hit] = val[order[safe_pos[hit]]]
    return out


def rank_margin_for_targets(
    model: SupplyGatedDRUM,
    targets: torch.Tensor,
    graph,
    args,
    k_per_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pass                                                                     
                                                                             
                                                                               
    raw = kernel_sparse_scores(model, targets[:, 0], targets[:, 1], graph, args, is_training=False)
    scores = bound_sparse_or_dense_scores(model, raw)
    n = int(targets.shape[0])
    k_per_target = k_per_target.long().clamp_min(1)
    if not scores.is_sparse:
        s_true = scores[torch.arange(n, device=targets.device), targets[:, 2].long()]
        with torch.no_grad():
            k_max = int(k_per_target.max().item())
            top = torch.topk(scores.detach(), min(k_max, scores.shape[1]), dim=1).values
            s_kth = top[torch.arange(n, device=top.device), (k_per_target - 1).clamp(max=top.shape[1] - 1)]
        return s_true, s_kth
    sc = scores.coalesce()
    idx, val = sc.indices(), sc.values()
    entity_count = int(sc.shape[1])
    keys = idx[0].long() * entity_count + idx[1].long()
    order = torch.argsort(keys)
    sorted_keys, sorted_val = keys[order], val[order]
    target_keys = torch.arange(n, device=targets.device, dtype=torch.long) * entity_count + targets[:, 2].long()
    pos = torch.searchsorted(sorted_keys, target_keys)
    in_bounds = pos < sorted_keys.numel()
    safe_pos = pos.clamp(max=max(int(sorted_keys.numel()) - 1, 0))
    hit = in_bounds & (sorted_keys[safe_pos] == target_keys)
    s_true = torch.zeros(n, dtype=val.dtype, device=val.device)
    if bool(hit.any().item()):
        s_true[hit] = val[order[safe_pos[hit]]]
    with torch.no_grad():
        rows = idx[0].long()
        s_kth = torch.zeros(n, dtype=val.dtype, device=val.device)
        vals_d = val.detach()
        for row in range(n):
            rv = vals_d[rows == row]
            k = int(k_per_target[row].item())
            if rv.numel() == 0:
                continue
            k = min(k, int(rv.numel()))
            s_kth[row] = torch.topk(rv, k).values[-1]
    return s_true, s_kth


def sparse_candidate_rows(scores: torch.Tensor) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    pass                                                               

                                                                                
                                                                             
                                                                              
                                                                            
                                                                           
       
    scores = scores.coalesce()
    idx = scores.indices()
    val = scores.values()
    if idx.numel() == 0:
        return {}
    rows_cpu = idx[0].long().detach().cpu().numpy()
    tails_cpu = idx[1].long().detach().cpu().numpy()
    values_cpu = val.detach().to(torch.float32).cpu().numpy()
    if rows_cpu.size > 1 and np.any(rows_cpu[1:] < rows_cpu[:-1]):
        order = np.argsort(rows_cpu, kind="stable")
        rows_cpu = rows_cpu[order]
        tails_cpu = tails_cpu[order]
        values_cpu = values_cpu[order]
    unique_rows, counts = np.unique(rows_cpu, return_counts=True)
    out = {}
    start = 0
    for row, count in zip(unique_rows.tolist(), counts.tolist()):
        end = start + int(count)
        out[int(row)] = (tails_cpu[start:end], values_cpu[start:end])
        start = end
    return out


__all__ = [
    "bound_sparse_or_dense_scores",
    "bounded_drum_scores",
    "concat_score_batches",
    "compact_kernel_graph_for_decode",
    "hard_supply_context",
    "kernel_sparse_scores",
    "masked_edge_context",
    "proof_for_targets",
    "proof_scores_for_targets",
    "proof_supply_context",
    "sparse_candidate_rows",
]
