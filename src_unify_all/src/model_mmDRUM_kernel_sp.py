import sys
import os
import torch
import numpy as np
import time
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fastlog_kernel.fastlog_kernel import (
    FastLogFunctionSparse3DTopK,
    build_csr_structure,
)
from utils import log_loss_common_sp


def _sparse_coo_unique(idx, val, shape):
    return torch.sparse_coo_tensor(idx, val, shape, is_coalesced=True)


def _model_stage_timing_enabled():
    return os.environ.get("FASTLOG_STAGE_TIMING", "") == "1"


def _model_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _model_log(stage, start_time):
    if not _model_stage_timing_enabled():
        return
    _model_sync()
    print("StageTiming mmDRUM_kernel_sp stage:{} seconds:{:.4f}".format(
        stage,
        time.perf_counter() - float(start_time),
    ))


def norm_sp_3d(s, detach_norm=False):
    s = s.coalesce()
    idx = s.indices()
    val = s.values()
    B, L, _ = s.shape
    denom = torch.zeros(B, L, device=val.device, dtype=val.dtype)
    denom.index_put_((idx[0], idx[1]), val, accumulate=True)
    denom = denom.clamp(min=1e-7)
    denom_for_grad = denom.detach() if detach_norm else denom
    val_norm = val / denom_for_grad[idx[0], idx[1]]
    keep = val_norm != 0
    idx = idx[:, keep]
    val_norm = val_norm[keep]
    return _sparse_coo_unique(idx, val_norm, s.shape)


def norm_sp_3d_pair(s):
    s = s.coalesce()
    idx = s.indices()
    val = s.values()
    B, L, _ = s.shape
    denom = torch.zeros(B, L, device=val.device, dtype=val.dtype)
    denom.index_put_((idx[0], idx[1]), val, accumulate=True)
    denom = denom.clamp(min=1e-7)
    val_base = val / denom[idx[0], idx[1]]
    val_adv = val / denom.detach()[idx[0], idx[1]]
    keep = (val_base != 0) | (val_adv != 0)
    idx = idx[:, keep]
    val_base = val_base[keep]
    val_adv = val_adv[keep]
    base = _sparse_coo_unique(idx, val_base, s.shape)
    adv = _sparse_coo_unique(idx, val_adv, s.shape)
    return base, adv


def dropout_sp_3d_pair(base, adv, dropout):
    p = float(getattr(dropout, "p", 0.0) or 0.0)
    if p <= 0.0:
        return base, adv
    keep = (torch.rand_like(base.values()) > p).to(base.values().dtype) / max(1.0 - p, 1e-12)
    idx = base.indices()
    base = _sparse_coo_unique(idx, base.values() * keep, base.shape)
    adv = _sparse_coo_unique(idx, adv.values() * keep.to(adv.values().dtype), adv.shape)
    return base, adv


def _max_sp_3d_from_values_pair(idx, val_base, val_adv, shape):
    B, slot_count, E = shape
    flat_key = idx[0] * E + idx[2]
    sort_key = flat_key * slot_count + idx[1]
    order = torch.argsort(sort_key)
    sorted_key = sort_key[order] // slot_count
    sorted_val = val_base.detach()[order]
    uniq, inv = torch.unique_consecutive(sorted_key, return_inverse=True)
    max_val = torch.full((uniq.shape[0],), float("-inf"), device=val_base.device, dtype=val_base.dtype)
    max_val.scatter_reduce_(0, inv, sorted_val, reduce="amax", include_self=True)
    sorted_pos = torch.arange(sorted_val.shape[0], device=val_base.device, dtype=torch.long)
    sentinel = sorted_val.shape[0]
    matched = sorted_val == max_val[inv]
    candidate_pos = torch.where(matched, sorted_pos, torch.full_like(sorted_pos, sentinel))
    arg_sorted_pos = torch.full((uniq.shape[0],), sentinel, device=val_base.device, dtype=torch.long)
    arg_sorted_pos.scatter_reduce_(0, inv, candidate_pos, reduce="amin", include_self=True)
    raw_arg = order[arg_sorted_pos]
    out_base = val_base[raw_arg]
    out_adv = val_adv[raw_arg]
    keep = (out_base != 0) | (out_adv != 0)
    uniq = uniq[keep]
    out_base = out_base[keep]
    out_adv = out_adv[keep]
    out_b = uniq // E
    out_e = uniq % E
    out_idx = torch.stack([out_b, out_e], dim=0)
    return (
        _sparse_coo_unique(out_idx, out_base, [B, E]),
        _sparse_coo_unique(out_idx, out_adv, [B, E]),
    )


def _max_sp_3d_from_values(idx, val, shape):
    B, slot_count, E = shape
    flat_key = idx[0] * E + idx[2]
    sort_key = flat_key * slot_count + idx[1]
    order = torch.argsort(sort_key)
    sorted_key = sort_key[order] // slot_count
    sorted_val = val.detach()[order]
    uniq, inv = torch.unique_consecutive(sorted_key, return_inverse=True)
    max_val = torch.full((uniq.shape[0],), float("-inf"), device=val.device, dtype=val.dtype)
    max_val.scatter_reduce_(0, inv, sorted_val, reduce="amax", include_self=True)
    sorted_pos = torch.arange(sorted_val.shape[0], device=val.device, dtype=torch.long)
    sentinel = sorted_val.shape[0]
    matched = sorted_val == max_val[inv]
    candidate_pos = torch.where(matched, sorted_pos, torch.full_like(sorted_pos, sentinel))
    arg_sorted_pos = torch.full((uniq.shape[0],), sentinel, device=val.device, dtype=torch.long)
    arg_sorted_pos.scatter_reduce_(0, inv, candidate_pos, reduce="amin", include_self=True)
    out_val = val[order[arg_sorted_pos]]
    keep = out_val != 0
    uniq = uniq[keep]
    out_val = out_val[keep]
    out_b = uniq // E
    out_e = uniq % E
    out_idx = torch.stack([out_b, out_e], dim=0)
    return _sparse_coo_unique(out_idx, out_val, [B, E])


def norm_sp_3d_pair_max(s, dropout=None):
    s = s.coalesce()
    idx = s.indices()
    val = s.values()
    B, L, _ = s.shape
    denom = torch.zeros(B, L, device=val.device, dtype=val.dtype)
    denom.index_put_((idx[0], idx[1]), val, accumulate=True)
    denom = denom.clamp(min=1e-7)
    val_base = val / denom[idx[0], idx[1]]
    val_adv = val / denom.detach()[idx[0], idx[1]]
    p = float(getattr(dropout, "p", 0.0) or 0.0) if dropout is not None else 0.0
    if p > 0.0:
        keep_mask = (torch.rand_like(val_base) > p).to(val_base.dtype) / max(1.0 - p, 1e-12)
        val_base = val_base * keep_mask
        val_adv = val_adv * keep_mask.to(val_adv.dtype)
    keep = (val_base != 0) | (val_adv != 0)
    idx = idx[:, keep]
    val_base = val_base[keep]
    val_adv = val_adv[keep]
    return _max_sp_3d_from_values_pair(idx, val_base, val_adv, s.shape)


def norm_sp_3d_max(s, dropout=None, detach_norm=False):
    s = s.coalesce()
    p = float(getattr(dropout, "p", 0.0) or 0.0) if dropout is not None else 0.0
    if p > 0.0:
        s = norm_sp_3d(s, detach_norm=detach_norm)
        s = _sparse_coo_unique(s.indices(), dropout(s.values()), s.shape)
        return max_sp_3d(s)
    idx = s.indices()
    val = s.values()
    B, L, _ = s.shape
    denom = torch.zeros(B, L, device=val.device, dtype=val.dtype)
    denom.index_put_((idx[0], idx[1]), val, accumulate=True)
    denom = denom.clamp(min=1e-7)
    denom_for_grad = denom.detach() if detach_norm else denom
    val_norm = val / denom_for_grad[idx[0], idx[1]]
    keep = val_norm != 0
    return _max_sp_3d_from_values(idx[:, keep], val_norm[keep], s.shape)


def max_sp_3d(s):
    s = s.coalesce()
    idx = s.indices()
    val = s.values()
    B, _, E = s.shape
    flat_key = idx[0] * E + idx[2]
    slot_count = s.shape[1]
    sort_key = flat_key * slot_count + idx[1]
    order = torch.argsort(sort_key)
    sorted_key = sort_key[order] // slot_count
    sorted_val = val.detach()[order]
    uniq, inv = torch.unique_consecutive(sorted_key, return_inverse=True)
    max_val = torch.full((uniq.shape[0],), float("-inf"), device=val.device, dtype=val.dtype)
    max_val.scatter_reduce_(0, inv, sorted_val, reduce="amax", include_self=True)
    sorted_pos = torch.arange(sorted_val.shape[0], device=val.device, dtype=torch.long)
    sentinel = sorted_val.shape[0]
    matched = sorted_val == max_val[inv]
    candidate_pos = torch.where(matched, sorted_pos, torch.full_like(sorted_pos, sentinel))
    arg_sorted_pos = torch.full((uniq.shape[0],), sentinel, device=val.device, dtype=torch.long)
    arg_sorted_pos.scatter_reduce_(0, inv, candidate_pos, reduce="amin", include_self=True)
    out_val = val[order[arg_sorted_pos]]
    keep = out_val != 0
    uniq = uniq[keep]
    out_val = out_val[keep]
    out_b = uniq // E
    out_e = uniq % E
    out_idx = torch.stack([out_b, out_e], dim=0)
    return _sparse_coo_unique(out_idx, out_val, [B, E])


def sum_sp_3d(s):
    s = s.coalesce()
    idx = s.indices()
    val = s.values()
    B, _, E = s.shape
    idx2 = torch.stack([idx[0], idx[2]], dim=0)
    return torch.sparse_coo_tensor(idx2, val, [B, E]).coalesce()


def sum_sparse3d_concat(*states):
    states = [s.coalesce() for s in states]
    non_empty = [s for s in states if s._nnz() > 0]
    if not non_empty:
        empty_i = torch.empty(0, device=states[0].device, dtype=torch.long)
        empty_v = torch.empty(0, device=states[0].device, dtype=states[0].values().dtype)
        return _sparse_coo_unique(torch.stack([empty_i, empty_i, empty_i], dim=0), empty_v, states[0].shape)
    if len(non_empty) == 1:
        return non_empty[0]
    idx = torch.cat([s.indices() for s in non_empty], dim=1)
    val = torch.cat([s.values() for s in non_empty], dim=0)
    return torch.sparse_coo_tensor(idx, val, non_empty[0].shape).coalesce()


def sparse3d_max(*states):
    states = [s.coalesce() for s in states]
    B, L, E = states[0].shape
    state_count = len(states)
    keys = []
    vals = []
    for i, s in enumerate(states):
        idx = s.indices()
        keys.append((idx[0] * (L * E) + idx[1] * E + idx[2]) * state_count + i)
        vals.append(s.values())
    sort_key = torch.cat(keys, dim=0)
    all_val = torch.cat(vals, dim=0)
    order = torch.argsort(sort_key)
    key = sort_key[order] // state_count
    sorted_val = all_val.detach()[order]
    uniq, inv = torch.unique_consecutive(key, return_inverse=True)
    max_val = torch.full((uniq.shape[0],), float("-inf"), device=all_val.device, dtype=all_val.dtype)
    max_val.scatter_reduce_(0, inv, sorted_val, reduce="amax", include_self=True)
    sorted_pos = torch.arange(sorted_val.shape[0], device=all_val.device, dtype=torch.long)
    sentinel = sorted_val.shape[0]
    matched = sorted_val == max_val[inv]
    candidate_pos = torch.where(matched, sorted_pos, torch.full_like(sorted_pos, sentinel))
    arg_sorted_pos = torch.full((uniq.shape[0],), sentinel, device=all_val.device, dtype=torch.long)
    arg_sorted_pos.scatter_reduce_(0, inv, candidate_pos, reduce="amin", include_self=True)
    out_val = all_val[order[arg_sorted_pos]]
    keep = out_val != 0
    uniq = uniq[keep]
    out_val = out_val[keep]
    out_b = uniq // (L * E)
    rem = uniq % (L * E)
    out_l = rem // E
    out_e = rem % E
    out_idx = torch.stack([out_b, out_l, out_e], dim=0)
    return _sparse_coo_unique(out_idx, out_val, [B, L, E])


class Model(nn.Module):
    def __init__(self, n, T, L, E, N, emb_size, tau_1=1, tau_2=1, use_gpu=False,
                 dropout=0.1, c=100000, use_soft=False, use_topk=False, topk_edges=0):
        super(Model, self).__init__()
        self.T = T
        self.L = L
        self.E = E
        self.n = n
        self.r_size = (self.n - 1) // 2
        self.N = N
        self.use_soft = use_soft
        self.use_topk = use_topk
        self.topk_edges = topk_edges
        self.emb_size = emb_size
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.c = c
        self.emb = nn.Parameter(torch.Tensor(self.n, self.emb_size))
        nn.init.kaiming_uniform_(self.emb, a=np.sqrt(5))
        self.lstm = nn.ModuleList()
        for _ in range(self.L):
            self.lstm.append(nn.LSTM(self.emb_size, self.emb_size, 1, bidirectional=True))
        self.linear = nn.Linear(2 * self.emb_size, self.n)
        if self.use_soft:
            self.reset_weight_param(torch.ones(self.N, 1))
        else:
            self.weight_param = None

        self.use_gpu = use_gpu
        self.dropout = nn.Dropout(dropout)
        self._csr_struct = None
        self._csr_struct_key = None
        self.transition_reduce = "sum"

    def reset_weight_param(self, new_weight):
        new_weight = new_weight.to(self.emb.device).clone()
        new_weight[new_weight == 1] = 2.5
        self.weight_param = nn.Parameter(new_weight)

    @property
    def weight(self):
        if self.use_soft:
            return torch.sigmoid(self.weight_param)
        return None

    def _ensure_csr_struct(self, e2triple, triple2e, r2triple):
        row_indices = e2triple[0]
        col_indices = triple2e[1]
        r_indices = r2triple[0]
        struct_key = (row_indices.data_ptr(), col_indices.data_ptr())
        if self._csr_struct is None or self._csr_struct_key != struct_key:
            self._csr_struct = build_csr_structure(
                row_indices, col_indices, r_indices, self.E
            )
            self._csr_struct_key = struct_key
        return self._csr_struct

    def forward(self, input_x, input_r, e2triple, triple2e, r2triple,
                is_training=False, input_y=None, csr_struct=None, mask_values=None,
                disable_edge_weight_grad=False, return_detach_norm_pair=False,
                detach_norm=False):
        t_stage = time.perf_counter()
        B = input_x.shape[0]
        L = self.L
        E = self.E
        device = input_x.device

        input_emb_ori = torch.index_select(self.emb, index=input_r, dim=0)
        input_emb = torch.stack([input_emb_ori] * (self.T + 1), dim=1)
        input_emb[:, -1, :] = self.emb[-1]
        input_emb = input_emb.transpose(1, 0)
        w_all = []
        for l in range(L):
            rnn_outputs, _ = self.lstm[l](input_emb)
            rnn_outputs = rnn_outputs.transpose(1, 0)
            outputs = self.linear(rnn_outputs[:, :-1, :])
            w_all.append(outputs)
        w_all = torch.stack(w_all, dim=2)
        _model_log("prepare_rule_logits", t_stage)

        t_stage = time.perf_counter()
        if csr_struct is not None:
            _csr_struct = csr_struct
        else:
            _csr_struct = self._ensure_csr_struct(e2triple, triple2e, r2triple)
        _model_log("prepare_csr_struct", t_stage)

        t_stage = time.perf_counter()
        if mask_values is not None:
            _mask_values = mask_values if mask_values.device == device else mask_values.to(device)
        else:
            _mask_values = e2triple[2].float().to(device)
        _model_log("prepare_mask_values", t_stage)

        t_stage = time.perf_counter()
        sp_batch = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(L)
        sp_level = torch.arange(L, device=device, dtype=torch.long).repeat(B)
        sp_entity = input_x.long().repeat_interleave(L)
        sp_value = torch.ones(B * L, device=device, dtype=torch.float)
        state = torch.sparse_coo_tensor(
            torch.stack([sp_batch, sp_level, sp_entity], dim=0),
            sp_value,
            [B, L, E],
            is_coalesced=True,
        )
        states = [state]
        final_out = None
        final_pair_out = None
        _model_log("build_initial_sparse_state", t_stage)

        (ori_row_ptr, ori_col_ind, ori_r_ind, order_ori,
         inv_row_ptr, inv_col_ind, inv_r_ind, order_inv,
         ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
         inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count) = _csr_struct

        for t in range(self.T):
            t_stage = time.perf_counter()
            w_probs = w_all[:, t, :, :]
            w_t = torch.softmax(w_probs / self.tau_1, dim=-1)
            topk_edges = self.topk_edges if self.use_topk else (1 << 60)
            use_max_reduce = str(getattr(self, "transition_reduce", "sum") or "sum") == "max"
            edge_weight_is_score = False
            if disable_edge_weight_grad and self.weight_param is not None:
                weight_arg = self.weight_param.detach()
            else:
                weight_arg = self.weight_param
            out_ori, out_inv, out_ind = FastLogFunctionSparse3DTopK.apply(
                states[-1].indices()[0], states[-1].indices()[1], states[-1].indices()[2], states[-1].values(), w_t,
                ori_row_ptr, ori_col_ind, ori_r_ind, order_ori,
                inv_row_ptr, inv_col_ind, inv_r_ind, order_inv,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                _mask_values, weight_arg, B, L, E, self.r_size, False, topk_edges, 1 if use_max_reduce else 0,
                edge_weight_is_score
            )
            _model_log("hop{}_kernel_forward".format(t), t_stage)
            t_stage = time.perf_counter()
            if use_max_reduce:
                s = sparse3d_max(out_ind, out_ori, out_inv)
            else:
                s = sum_sparse3d_concat(out_ori, out_inv, out_ind)
            if return_detach_norm_pair and t == self.T - 1:
                if use_max_reduce:
                    final_pair_out = norm_sp_3d_pair_max(s, self.dropout if is_training else None)
                    states.append(None)
                    state_adv = None
                else:
                    s_base, s_adv = norm_sp_3d_pair(s)
                    if is_training:
                        s_base, s_adv = dropout_sp_3d_pair(s_base, s_adv, self.dropout)
                    states.append(s_base)
                    state_adv = s_adv
            else:
                if use_max_reduce and t == self.T - 1:
                    final_out = norm_sp_3d_max(s, self.dropout if is_training else None, detach_norm=detach_norm)
                    states.append(None)
                else:
                    s = norm_sp_3d(s, detach_norm=detach_norm)
                    if is_training:
                        s = _sparse_coo_unique(s.indices(), self.dropout(s.values()), s.shape)
                    states.append(s)
            _model_log("hop{}_postprocess".format(t), t_stage)

        if final_pair_out is not None:
            return final_pair_out
        if final_out is not None:
            return final_out

        t_stage = time.perf_counter()
        if str(getattr(self, "transition_reduce", "sum") or "sum") == "max":
            out = max_sp_3d(states[-1])
        else:
            out = sum_sp_3d(states[-1])
        if return_detach_norm_pair:
            if str(getattr(self, "transition_reduce", "sum") or "sum") == "max":
                out_adv = max_sp_3d(state_adv)
            else:
                out_adv = sum_sp_3d(state_adv)
            _model_log("final_reduce", t_stage)
            return out, out_adv
        _model_log("final_reduce", t_stage)
        return out

    def log_loss(self, p_score, label, thr=1e-7):
        return log_loss_common_sp(p_score, label, self.E, self.tau_2)
