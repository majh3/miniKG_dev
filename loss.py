

                                                        

                                                 
                                   
                                           
                                                                          
                            
   

from __future__ import annotations

import json
from types import SimpleNamespace

import torch
import torch.nn.functional as F

try:
    from .drum import SupplyGatedDRUM
    from .facts import encode_membership_keys, fact_membership_from_sorted_keys
    from .proof import (
        bound_sparse_scores,
        clear_cached_train_target_edge_scores,
        kernel_sparse_scores,
        masked_edge_context,
        proof_graph_masked_edge_context,
        proof_scores_for_targets,
        proof_supply_context,
    )
    from .graph import Graph
    from .negative import relation_subset_tensor_for_targets, rule_generated_negative_targets, sample_corrupt_targets
except ImportError:                           
    from drum import SupplyGatedDRUM
    from facts import encode_membership_keys, fact_membership_from_sorted_keys
    from proof import (
        bound_sparse_scores,
        clear_cached_train_target_edge_scores,
        kernel_sparse_scores,
        masked_edge_context,
        proof_graph_masked_edge_context,
        proof_scores_for_targets,
        proof_supply_context,
    )
    from graph import Graph
    from negative import relation_subset_tensor_for_targets, rule_generated_negative_targets, sample_corrupt_targets


def orient_targets_for_query(targets: torch.Tensor, args: SimpleNamespace) -> torch.Tensor:
    inverse = getattr(args, "_query_orientation_inverse_tensor", None)
    if inverse is None:
        return targets
    mask = inverse[targets[:, 1].long()]
    if not bool(mask.any().item()):
        return targets
    out = targets.clone()
    out[mask, 0] = targets[mask, 2]
    out[mask, 2] = targets[mask, 0]
    return out


def orient_false_targets_for_query(targets: torch.Tensor, args: SimpleNamespace) -> torch.Tensor:
    return orient_targets_for_query(targets, args)


                                                                                 
                                                                             
                                                                                    
                                      

@torch.no_grad()
def _witness_probe_metrics(
    gradient: torch.Tensor,
    batch_idx: torch.Tensor,
    gate_local: bool,
) -> tuple[float, int, float]:
    pass                                                                     
    detached = gradient.detach()
    leak = -1.0
    if detached.is_sparse:
        sparse = detached.coalesce()
        values_abs = sparse.values().abs()
        absmax = float(values_abs.max().item()) if int(values_abs.numel()) > 0 else 0.0
        nnz = int(torch.count_nonzero(values_abs).item())
        if not gate_local and int(batch_idx.numel()) > 0:
            leak = 0.0
            entries = int(sparse._nnz())
            if entries > 0:
                rows = sparse.indices()[0].long()
                entry_absmax = values_abs.reshape(entries, -1).amax(dim=1)
                batch = torch.unique(batch_idx.view(-1).long(), sorted=True)
                positions = torch.searchsorted(batch, rows)
                in_bounds = positions < int(batch.numel())
                safe = positions.clamp(max=max(int(batch.numel()) - 1, 0))
                hits = in_bounds & (batch[safe] == rows)
                if bool(hits.any().item()):
                    leak = float(entry_absmax[hits].max().item())
        return absmax, nnz, leak

    values_abs = detached.abs().view(-1)
    absmax = float(values_abs.max().item()) if int(values_abs.numel()) > 0 else 0.0
    nnz = int(torch.count_nonzero(values_abs).item())
    if not gate_local and int(batch_idx.numel()) > 0:
        leak = float(values_abs[batch_idx.view(-1).long()].max().item())
    return absmax, nnz, leak


def _sparse_gate_rows_values(gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pass                                                                      
    sparse = gradient.coalesce()
    if sparse.dim() != 2 or int(sparse.shape[1]) != 1:
        raise RuntimeError(f"expected [N, 1] sparse gate gradient, got {tuple(sparse.shape)}")
    if sparse.sparse_dim() == 1 and sparse.dense_dim() == 1:
        rows = sparse.indices()[0].long()
        values = sparse.values().reshape(-1)
    elif sparse.sparse_dim() == 2 and sparse.dense_dim() == 0:
        indices = sparse.indices()
        if int(indices.shape[1]) > 0 and bool((indices[1] != 0).any().item()):
            raise RuntimeError("gate gradient contains a nonzero column index")
        rows = indices[0].long()
        values = sparse.values().reshape(-1)
    else:
        raise RuntimeError(
            f"unsupported sparse gate layout: sparse_dim={sparse.sparse_dim()} dense_dim={sparse.dense_dim()}"
        )
    return rows, values


def _sparse_parameter_dot(parameter: torch.Tensor, coefficient: torch.Tensor) -> torch.Tensor:
    pass                                                                    
    if not coefficient.is_sparse:
        return (parameter * coefficient).sum()
    rows, values = _sparse_gate_rows_values(coefficient)
    selected = F.embedding(rows, parameter, sparse=True).view(-1)
    return (selected * values.detach()).sum()


@torch.no_grad()
def _update_witness_ema_(
    cache: torch.Tensor,
    gradient: torch.Tensor,
    beta: float,
    touch_clocked: bool = False,
) -> None:
    pass                                                               
    if touch_clocked:
        if gradient.is_sparse:
            rows, values = _sparse_gate_rows_values(gradient)
            touched = values != 0
            rows = rows[touched].to(device=cache.device)
            values = values[touched].to(device=cache.device, dtype=cache.dtype)
            if int(rows.numel()) > 0:
                flat = cache.view(-1)
                flat[rows] = flat[rows] * (1.0 - beta) + values * beta
        else:
            gradient = gradient.to(device=cache.device, dtype=cache.dtype)
            touched = gradient != 0
            cache[touched] = cache[touched] * (1.0 - beta) + gradient[touched] * beta
        return
    cache.mul_(1.0 - beta)
    if gradient.is_sparse:
        rows, values = _sparse_gate_rows_values(gradient)
        rows = rows.to(device=cache.device)
        values = values.to(device=cache.device, dtype=cache.dtype)
        if int(rows.numel()) > 0:
            cache.view(-1).index_add_(0, rows, values, alpha=beta)
    else:
        cache.add_(gradient.to(device=cache.device, dtype=cache.dtype), alpha=beta)


@torch.no_grad()
def apply_chunked_nbe_gate_update(
    model: SupplyGatedDRUM,
    args: SimpleNamespace,
    optimizers: list,
) -> dict:
    pass                                                                        
    pending = getattr(args, "_nbe_dense_update_pending", None)
    args._nbe_dense_update_pending = None
    if not bool(getattr(args, "_nbe_chunked_dense_update", False)) or pending is None:
        return {"nbe_chunked_dense_update": False}

    if model.weight_override is not None or model.weight_transform is not None or model.edge_weight_mask is not None:
        raise RuntimeError("chunked NBE update requires the unmodified soft gate-logit parameterization")

    matches = []
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            if any(parameter is model.weight_param for parameter in group["params"]):
                matches.append((optimizer, group))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one gate optimizer group, found {len(matches)}")
    gate_optimizer, gate_group = matches[0]
    if not isinstance(gate_optimizer, torch.optim.SGD):
        raise RuntimeError("chunked NBE update is exactly compatible only with plain SGD")
    incompatible = (
        float(gate_group.get("momentum", 0.0)) != 0.0
        or float(gate_group.get("weight_decay", 0.0)) != 0.0
        or bool(gate_group.get("nesterov", False))
        or bool(gate_group.get("maximize", False))
    )
    if incompatible:
        raise RuntimeError("chunked NBE update requires SGD without momentum, weight decay, nesterov, or maximize")

    sparse_grad = model.weight_param.grad
    dense_grad_flat = None
    if sparse_grad is None:
        sparse_rows = torch.empty(0, dtype=torch.long, device=model.weight_param.device)
        sparse_values = model.weight_param.new_empty(0)
    elif sparse_grad.is_sparse:
        sparse_rows, sparse_values = _sparse_gate_rows_values(sparse_grad)
    else:
                                                                       
                                                                        
                                                                          
                                                                           
        dense_grad_flat = sparse_grad.detach().view(-1)
        sparse_rows = torch.empty(0, dtype=torch.long, device=model.weight_param.device)
        sparse_values = model.weight_param.new_empty(0)
        if not bool(getattr(args, "_nbe_dense_grad_warned", False)):
            args._nbe_dense_grad_warned = True
            print(
                json.dumps(
                    {
                        "event": "nbe_dense_gate_grad_fallback",
                        "n": int(dense_grad_flat.numel()),
                        "bytes": int(dense_grad_flat.numel() * dense_grad_flat.element_size()),
                        "note": "chunk-adding dense autograd gate grad; root densifier still open",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    witness_gain = float(pending.get("witness_gain", 0.0))
    pressure_on = bool(pending.get("pressure_on", False))
    st_gain = float(pending.get("st_gain", 0.0))
    gamma = float(pending.get("integrality_gamma", 0.0))
    c_cache = getattr(args, "_nbe_c_cache", None)
    w_cache = getattr(args, "_nbe_w_cache", None)
    if witness_gain != 0.0 and w_cache is None:
        raise RuntimeError("witness replay requested without a witness cache")
    if (pressure_on or st_gain != 0.0 or gamma != 0.0) and c_cache is None:
        raise RuntimeError("dense NBE pressure requested without a recoverability cache")

    chunk_size = max(1, int(getattr(args, "_nbe_dense_update_chunk_size", 1_000_000)))
    row_ranges: dict = {}
    if int(sparse_rows.numel()) > 0:
        row_chunks = torch.div(sparse_rows, chunk_size, rounding_mode="floor")
        chunk_ids, chunk_counts = torch.unique_consecutive(row_chunks, return_counts=True)
        cursor = 0
        for chunk_id, count in zip(chunk_ids.cpu().tolist(), chunk_counts.cpu().tolist()):
            row_ranges[int(chunk_id)] = (cursor, cursor + int(count))
            cursor += int(count)

    theta = model.weight_param.view(-1)
    tau = max(float(model.supply_temperature), 1e-12)
    lr = float(gate_group["lr"])
    for start in range(0, int(theta.numel()), chunk_size):
        end = min(start + chunk_size, int(theta.numel()))
        theta_chunk = theta[start:end]
        total_grad = torch.zeros_like(theta_chunk)
        w_chunk = w_cache.view(-1)[start:end] if w_cache is not None else None
        if w_chunk is not None and w_chunk.device != theta_chunk.device:
            w_chunk = w_chunk.to(device=theta_chunk.device, dtype=theta_chunk.dtype)
        if witness_gain != 0.0:
            total_grad.add_(w_chunk, alpha=-witness_gain)

        if pressure_on or st_gain != 0.0 or gamma != 0.0:
            c_raw = c_cache.view(-1)[start:end]
            measured = c_raw >= 0
            c_chunk = torch.where(measured, c_raw, torch.zeros_like(c_raw))
            sigma = None
            sigma_prime = None
            if pressure_on or gamma != 0.0:
                sigma = torch.sigmoid(theta_chunk / tau)
                sigma_prime = sigma * (1.0 - sigma) / tau
            if pressure_on:
                total_grad.add_(c_chunk * sigma_prime)
            if st_gain != 0.0:
                st_mask = measured if w_chunk is None else measured & (w_chunk <= 0)
                total_grad.add_(torch.where(st_mask, c_raw, torch.zeros_like(c_raw)), alpha=st_gain)
            if gamma != 0.0:
                integrality_grad = (1.0 - 2.0 * sigma) * sigma_prime
                total_grad.add_(torch.where(measured, integrality_grad, torch.zeros_like(integrality_grad)), alpha=gamma)

        sparse_range = row_ranges.get(start // chunk_size)
        if sparse_range is not None:
            left, right = sparse_range
            total_grad.index_add_(0, sparse_rows[left:right] - start, sparse_values[left:right])
        if dense_grad_flat is not None:
            total_grad.add_(dense_grad_flat[start:end])
        theta_chunk.add_(total_grad, alpha=-lr)

    model.weight_param.grad = None
    clear_cached_train_target_edge_scores(model)
    return {
        "nbe_chunked_dense_update": True,
        "nbe_chunk_size": int(chunk_size),
        "nbe_sparse_gate_nnz": int(sparse_rows.numel()),
        "nbe_dense_gate_grad": dense_grad_flat is not None,
        "nbe_gate_lr": float(lr),
    }


def false_sample_count(args: SimpleNamespace, candidate_count: int) -> int:
    requested = int(getattr(args, "_tnb_false_main_batch_count", 0) or 0)
    return min(max(0, requested), int(candidate_count))


def false_scores_for_targets(
    model: SupplyGatedDRUM,
    false_targets: torch.Tensor,
    graph: Graph,
    args: SimpleNamespace,
) -> torch.Tensor:
    if false_targets.numel() == 0:
        return torch.zeros((), dtype=model.weight_param.dtype, device=model.weight_param.device)
    with proof_supply_context(model, args):
        scores = proof_scores_for_targets(model, false_targets, graph, args, True)
    return scores.mean()


def indexed_soft_supply(model: SupplyGatedDRUM, batch_idx: torch.Tensor) -> torch.Tensor:
    logits = model.gate_logits(batch_idx)
    return torch.sigmoid(logits / model.supply_temperature)








def gate_local_positions(model: SupplyGatedDRUM, global_idx: torch.Tensor) -> torch.Tensor:
    if str(getattr(model, "gate_scope", "all")) == "all":
        return global_idx.long().view(-1)
    gate_global_index = model.gate_global_index
    flat_idx = global_idx.long().view(-1)
    pos = torch.searchsorted(gate_global_index, flat_idx)
    pos = pos.clamp(max=int(gate_global_index.numel()) - 1)
    hit = gate_global_index[pos] == flat_idx
    if not bool(hit.all().item()):
        missing = int((~hit).sum().item())
        raise ValueError(f"netbenefit cache touched facts outside trainable gate scope: {missing}")
    return pos


def source_credit_aux_from_proof(
    model: SupplyGatedDRUM,
    target_supply: torch.Tensor,
    proof_without_self: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    pass                                                                     
    credit_weight = (1.0 - target_supply).detach()
    active = int((credit_weight.detach() > 0).sum().item())
    weight_sum = credit_weight.sum()
    proof_mean = proof_without_self.detach().mean()
    if not bool((weight_sum > 0).detach().item()):
        zero = torch.zeros((), dtype=model.weight_param.dtype, device=model.weight_param.device)
        return zero, proof_mean, active
    loss = -(credit_weight * torch.log(proof_without_self.clamp_min(1e-8))).sum() / weight_sum.clamp_min(1e-8)
    return loss, proof_mean, active


def build_false_targets(
    model: SupplyGatedDRUM,
    facts_tensor: torch.Tensor,
    batch_idx: torch.Tensor,
    targets: torch.Tensor,
    truth_by_query: dict,
    entity_count: int,
    relation_count: int,
    sorted_true_keys: torch.Tensor,
    args: SimpleNamespace,
    generator: torch.Generator,
) -> torch.Tensor:
    extra_n = false_sample_count(args, facts_tensor.shape[0])
    if extra_n <= 0:
        return torch.empty((0, 3), dtype=facts_tensor.dtype, device=facts_tensor.device)

    extra_idx = torch.randint(0, facts_tensor.shape[0], (extra_n,), generator=generator, device=facts_tensor.device)

    if args.dataset == "family":
        false_relation_ids = relation_subset_tensor_for_targets(
            "",
            relation_count,
            facts_tensor.device,
            targets,
        )
        false_targets = rule_generated_negative_targets(
            model,
            facts_tensor,
            extra_idx,
            entity_count,
            relation_count,
            sorted_true_keys,
            "auto",
            false_relation_ids,
            1,
        )
        return orient_false_targets_for_query(false_targets, args)

    extra_targets = orient_targets_for_query(facts_tensor[extra_idx], args)
    return sample_corrupt_targets(extra_targets, truth_by_query, entity_count, generator)


def target_net_benefit_loss_final(
    model: SupplyGatedDRUM,
    facts_tensor: torch.Tensor,
    batch_idx: torch.Tensor,
    graph: Graph,
    truth_by_query: dict,
    entity_count: int,
    relation_count: int,
    sorted_true_keys: torch.Tensor,
    args: SimpleNamespace,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_targets = facts_tensor[batch_idx]
    targets = orient_targets_for_query(raw_targets, args)
    target_supply = indexed_soft_supply(model, batch_idx)
    args._source_credit_aux_precomputed = None
    with proof_supply_context(model, args):
        with proof_graph_masked_edge_context(model, batch_idx, args):
                true_covered = proof_scores_for_targets(model, targets, graph, args, True)
    if True:
                                                                             
                                                                        
                                                                             
                                                                               
                                                                                   
                                                                             
                                                                                
                                                                               
                                                                                  
        supply_live = indexed_soft_supply(model, batch_idx)
        c_econ = true_covered
        cache = getattr(args, "_nbe_c_cache", None)
        if False:
                                                                             
                                                                               
                                                                              
                                                                              
                                                                                
            jit = getattr(args, "_nbe_jitter_cache", None)
            if jit is None or jit.numel() != cache.numel():
                g = torch.Generator()
                g.manual_seed(20260705)
                jit = torch.rand(cache.numel(), generator=g) * 0.0
                jit = jit.to(cache.device)
                args._nbe_jitter_cache = jit
            live_idx = (
                gate_local_positions(model, batch_idx)
                if bool(getattr(args, "_nbe_c_cache_is_gate_local", False))
                else batch_idx
            )
            c_econ = (c_econ * (1.0 + jit[live_idx].view(c_econ.shape))).clamp(0.0, 1.0)
        keep_cost = supply_live + (1.0 - supply_live) * (1.0 - c_econ)
                                                              
        keep_cost = keep_cost.detach() + (keep_cost - keep_cost.detach()) * (
            0.5 * float(keep_cost.numel())
        )
                                                                                  
        rule_grad = (4.0 if args.dataset == "family" else 16.0) * true_covered
        inferred_true = -keep_cost + rule_grad - rule_grad.detach()
        if False:
                                                                            
                                                                           
                                                                          
                                                                        
                                                                         
                                                                              
            rule_grad_sum = rule_grad.sum()
            g_rf = (
                torch.autograd.grad(rule_grad_sum, model.weight_param, retain_graph=True, allow_unused=True)[0]
                if rule_grad_sum.requires_grad
                else None
            )
            if g_rf is not None:
                if g_rf.is_sparse:
                    rf_cancel = _sparse_parameter_dot(model.weight_param, g_rf.detach())
                else:
                    rf_cancel = (model.weight_param * g_rf.detach()).sum()
                inferred_true = inferred_true - (rf_cancel - rf_cancel.detach())
        chunked_dense_update = bool(getattr(args, "_nbe_chunked_dense_update", False))
        dense_update_pending = {
            "witness_gain": 0.0,
            "pressure_on": False,
            "st_gain": 0.0,
            "integrality_gamma": 0.0,
        }
        if True:
                                                                            
                                                                              
                                                                              
                                                                             
                                                                           
                                                                          
                                                                            
                                                                              
                                                                               
            c_credit = c_econ.view(-1)
            credit_sum = ((1.0 - supply_live).detach().view(-1) * c_credit).sum()
                                                                             
                                                                           
                                                                             
                                                                              
                                                                            
                                                                           
            gw = (
                torch.autograd.grad(credit_sum, model.weight_param, retain_graph=True, allow_unused=True)[0]
                if credit_sum.requires_grad
                else None
            )
            if gw is not None:
                                                                        
                                                                            
                                                                                
                                                                       
                w_scale = float(facts_tensor.shape[0]) / float(max(int(batch_idx.numel()), 1))
                if gw.is_sparse:
                    gw = gw.coalesce()
                    gw = torch.sparse_coo_tensor(
                        gw.indices(), gw.values() * w_scale, gw.size(),
                        device=gw.device, dtype=gw.dtype,
                    ).coalesce()
                else:
                    gw = gw.mul_(w_scale)
                if not bool(getattr(args, "_nbe_w_probe_done", False)):
                    args._nbe_w_probe_done = True
                    absmax, nnz, leak = _witness_probe_metrics(
                        gw,
                        batch_idx,
                        bool(getattr(args, "_nbe_c_cache_is_gate_local", False)),
                    )
                    print(f"[WITNESS_PROBE] absmax={absmax:.3e} nnz={nnz} batch_leak={leak:.3e}", flush=True)
                w_cache = getattr(args, "_nbe_w_cache", None)
                if w_cache is None or w_cache.shape != model.weight_param.shape:
                                                                          
                                                                             
                                                                      
                    cache_device = (
                        torch.device("cpu")
                        if chunked_dense_update and model.weight_param.device.type == "cuda"
                        else model.weight_param.device
                    )
                    w_cache = torch.zeros(
                        model.weight_param.shape,
                        dtype=model.weight_param.dtype,
                        device=cache_device,
                    )
                    args._nbe_w_cache = w_cache
                    print(
                        json.dumps(
                            {
                                "event": "nbe_witness_cache_placement",
                                "device": str(w_cache.device),
                                "bytes": int(w_cache.numel() * w_cache.element_size()),
                                "gpu_resident_during_proof": w_cache.device.type == "cuda",
                                "equivalence": "same_fp32_ema_chunk_copied_for_sgd_update",
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                _update_witness_ema_(
                    w_cache,
                    gw,
                    0.1,
                    touch_clocked=False,
                )
                if chunked_dense_update:
                    dense_update_pending["witness_gain"] = 1.0
                else:
                                                                                
                    w_replay = (model.weight_param * w_cache.detach().to(model.weight_param.device)).sum()
                    inferred_true = inferred_true + (w_replay - w_replay.detach())
        if cache is not None:
            cache_idx = (
                gate_local_positions(model, batch_idx)
                if bool(getattr(args, "_nbe_c_cache_is_gate_local", False))
                else batch_idx
            )
                                                                              
                                                                            
                                                                            
                                                                              
                                                        
            with torch.no_grad():
                old = cache[cache_idx]
                fresh = c_econ.detach().view(-1)
                cache[cache_idx] = torch.where(old >= 0, 0.5 * old + (1.0 - 0.5) * fresh, fresh)
            if int(cache_idx.numel()) > 0:
                args._nbe_has_measured = True
            has_measured = bool(getattr(args, "_nbe_has_measured", False))
            measured = None if chunked_dense_update else (cache >= 0)
            if chunked_dense_update:
                dense_update_pending["pressure_on"] = bool(getattr(args, "_nbe_pressure_on", True)) and has_measured
            if not chunked_dense_update and bool(getattr(args, "_nbe_pressure_on", True)) and bool(measured.any().item()):
                if bool(getattr(args, "_nbe_c_cache_is_gate_local", False)):
                    sigma_all = torch.sigmoid(model.weight_param.view(-1)[measured] / model.supply_temperature)
                else:
                    sigma_all = model.weight.view(-1)[measured]
                c_meas = cache[measured]
                dense_cost = (sigma_all + (1.0 - sigma_all) * (1.0 - c_meas)).sum()
                inferred_true = inferred_true - (dense_cost - dense_cost.detach())
            st_gain = float(getattr(args, "netbenefit_st_pressure_gain", 0.0))
            if chunked_dense_update and st_gain > 0.0 and has_measured:
                args._nbe_st_calls = int(getattr(args, "_nbe_st_calls", 0)) + 1
                if args._nbe_st_calls > 20:
                    dense_update_pending["st_gain"] = st_gain
            if not chunked_dense_update and st_gain > 0.0 and bool(measured.any().item()):
                args._nbe_st_calls = int(getattr(args, "_nbe_st_calls", 0)) + 1
                if args._nbe_st_calls > 20:
                    wc_st = getattr(args, "_nbe_w_cache", None)
                    with torch.no_grad():
                        st_mask = measured.clone()
                        if wc_st is not None:
                            st_mask &= ~(wc_st.view(-1).to(measured.device) > 0)
                        st_coef = torch.zeros_like(cache)
                        st_coef[st_mask] = cache[st_mask]
                    st_cost = st_gain * (model.weight_param.view(-1) * st_coef).sum()
                    inferred_true = inferred_true - (st_cost - st_cost.detach())
        if chunked_dense_update:
            active_dense_update = (
                float(dense_update_pending["witness_gain"]) != 0.0
                or bool(dense_update_pending["pressure_on"])
                or float(dense_update_pending["st_gain"]) != 0.0
                or float(dense_update_pending["integrality_gamma"]) != 0.0
            )
            args._nbe_dense_update_pending = dense_update_pending if active_dense_update else None
    true_loss = -inferred_true.mean()
    tnb_true_loss = true_loss
    can_reuse_aux_proof = (
        targets is raw_targets
        and model.step >= 1
        and getattr(args, "_tnb_proof_supply_override", None) is None
    )
    if can_reuse_aux_proof:
        aux_loss, aux_proof, aux_active = source_credit_aux_from_proof(model, target_supply, true_covered)
        true_loss = true_loss + 0.5 * aux_loss
        args._source_credit_aux_precomputed = (
            batch_idx,
            aux_loss.detach(),
            aux_proof,
            aux_active,
        )
    true_loss.backward()
    clear_cached_train_target_edge_scores(model)

    false_targets = build_false_targets(
        model,
        facts_tensor,
        batch_idx,
        targets,
        truth_by_query,
        entity_count,
        relation_count,
        sorted_true_keys,
        args,
        generator,
    )
    wrong_mean = false_scores_for_targets(model, false_targets, graph, args)
    clear_cached_train_target_edge_scores(model)
    return tnb_true_loss.detach(), inferred_true.detach().mean(), wrong_mean.detach()


def source_credit_aux_loss_final(
    model: SupplyGatedDRUM,
    facts_tensor: torch.Tensor,
    batch_idx: torch.Tensor,
    graph: Graph,
    args: SimpleNamespace,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if model.step < 1:
                                                                                  
                                                                                   
        zero = torch.zeros((), dtype=model.weight_param.dtype, device=model.weight_param.device)
        return zero, zero, 0

    precomputed = getattr(args, "_source_credit_aux_precomputed", None)
    args._source_credit_aux_precomputed = None
    if precomputed is not None and precomputed[0] is batch_idx:
        return precomputed[1], precomputed[2], precomputed[3]

    targets = facts_tensor[batch_idx]
    target_supply = indexed_soft_supply(model, batch_idx)
    with masked_edge_context(model, batch_idx):
        proof_without_self = proof_scores_for_targets(
            model, targets, graph, args, True
        )

    return source_credit_aux_from_proof(model, target_supply, proof_without_self)
