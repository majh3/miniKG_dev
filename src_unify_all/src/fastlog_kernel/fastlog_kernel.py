import torch
from torch import autograd
from torch.utils import cpp_extension
import os
import sys
import subprocess
import time


def load_fastlog_kernel():
    source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")
    sources = [os.path.join(source_dir, "fastlog.cpp")]

                                  
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        sources.append(os.path.join(source_dir, "fastlog.cu"))

    extra_cflags = ["-O3", "-std=c++17"]
    extra_cuda_cflags = ["-O3", "--use_fast_math"]

    if cuda_available:
        extra_cflags.append("-DWITH_CUDA")

    if sys.platform == 'darwin':
        sdk_path = subprocess.check_output(['xcrun', '--show-sdk-path']).decode().strip()
        cxx_include = os.path.join(sdk_path, 'usr', 'include', 'c++', 'v1')
        extra_cflags += ["-stdlib=libc++", f"-isysroot{sdk_path}", f"-I{cxx_include}"]

    if torch.backends.openmp.is_available() and not sys.platform.startswith('darwin'):
        extra_cflags += ["-fopenmp", "-DAT_PARALLEL_OPENMP"]
    else:
        extra_cflags.append("-DAT_PARALLEL_NATIVE")

    print("Loading FastLog kernel extension. This may take a while...")
    module = cpp_extension.load(
        name="fastlog_kernel_cpp",
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags if cuda_available else None,
        verbose=False
    )
    print("FastLog kernel loaded.")
    return module


_module = None


def _kernel_stage_timing_enabled():
    return os.environ.get("FASTLOG_STAGE_TIMING", "") == "1"


def _kernel_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _kernel_log(stage, start_time):
    if not _kernel_stage_timing_enabled():
        return
    _kernel_sync()
    print("StageTiming fastlog_kernel stage:{} seconds:{:.4f}".format(
        stage,
        time.perf_counter() - float(start_time),
    ))

def get_module():
    global _module
    if _module is None:
        t0 = time.perf_counter()
        _module = load_fastlog_kernel()
        _kernel_log("load_module", t0)
    return _module


def score_function(x):
    return torch.sigmoid(x)


def _coalesce_sparse3d_reduce(b, l, e, v, B, L, E, reduce="sum"):
    if v.numel() == 0:
        empty_i = torch.empty(0, device=v.device, dtype=torch.long)
        empty_v = torch.empty(0, device=v.device, dtype=v.dtype)
        idx = torch.stack([empty_i, empty_i, empty_i], dim=0)
        sp = torch.sparse_coo_tensor(idx, empty_v, [B, L, E]).coalesce()
        return sp, torch.empty(0, device=v.device, dtype=torch.long), torch.empty(0, device=v.device, dtype=v.dtype)

    keys = b * (L * E) + l * E + e
    sort_idx = torch.argsort(keys)
    sorted_keys = keys[sort_idx]
    sorted_values = v[sort_idx]
    unique_keys, inverse = torch.unique_consecutive(sorted_keys, return_inverse=True)
    out_values = torch.zeros(unique_keys.shape[0], device=v.device, dtype=v.dtype)
    if reduce == "max":
        out_values.scatter_reduce_(0, inverse, sorted_values, reduce="amax", include_self=False)
    else:
        out_values.scatter_add_(0, inverse, sorted_values)
    out_b = unique_keys // (L * E)
    rem = unique_keys % (L * E)
    out_l = rem // E
    out_e = rem % E
    sp = torch.sparse_coo_tensor(torch.stack([out_b, out_l, out_e], dim=0), out_values, [B, L, E]).coalesce()
    return sp, unique_keys, out_values


def _coalesce_sparse3d_max_with_arg(b, l, e, v, B, L, E, tie_break=None):
    if v.numel() == 0:
        empty_i = torch.empty(0, device=v.device, dtype=torch.long)
        empty_v = torch.empty(0, device=v.device, dtype=v.dtype)
        idx = torch.stack([empty_i, empty_i, empty_i], dim=0)
        sp = torch.sparse_coo_tensor(idx, empty_v, [B, L, E]).coalesce()
        return sp, empty_i, empty_v, empty_i

    keys = b * (L * E) + l * E + e
    sort_idx = torch.argsort(keys)
    sorted_keys = keys[sort_idx]
    sorted_values = v[sort_idx]
    if tie_break is not None:
        sorted_tie = tie_break.long()[sort_idx]
    else:
        sorted_tie = torch.arange(sorted_values.shape[0], device=v.device, dtype=torch.long)
    unique_keys, inverse = torch.unique_consecutive(sorted_keys, return_inverse=True)

    out_values = torch.full((unique_keys.shape[0],), float("-inf"), device=v.device, dtype=v.dtype)
    out_values.scatter_reduce_(0, inverse, sorted_values, reduce="amax", include_self=True)

    sorted_pos = torch.arange(sorted_values.shape[0], device=v.device, dtype=torch.long)
    sentinel = sorted_values.shape[0]
    matched = sorted_values == out_values[inverse]
    tie_sentinel = torch.iinfo(torch.long).max
    candidate_tie = torch.where(matched, sorted_tie, torch.full_like(sorted_tie, tie_sentinel))
    min_tie = torch.full((unique_keys.shape[0],), tie_sentinel, device=v.device, dtype=torch.long)
    min_tie.scatter_reduce_(0, inverse, candidate_tie, reduce="amin", include_self=True)
    candidate = torch.where(
        matched & (sorted_tie == min_tie[inverse]),
        sorted_pos,
        torch.full_like(sorted_pos, sentinel),
    )
    arg_sorted_pos = torch.full((unique_keys.shape[0],), sentinel, device=v.device, dtype=torch.long)
    arg_sorted_pos.scatter_reduce_(0, inverse, candidate, reduce="amin", include_self=True)
    raw_arg = sort_idx[arg_sorted_pos.clamp(max=max(sorted_values.shape[0] - 1, 0))]
    raw_arg = torch.where(arg_sorted_pos < sentinel, raw_arg, torch.full_like(raw_arg, -1))

    out_b = unique_keys // (L * E)
    rem = unique_keys % (L * E)
    out_l = rem // E
    out_e = rem % E
    sp = torch.sparse_coo_tensor(torch.stack([out_b, out_l, out_e], dim=0), out_values, [B, L, E]).coalesce()
    return sp, unique_keys, out_values, raw_arg


def _align_sparse_grad_to_raw_argmax(grad_sp, out_keys, raw_arg, raw_size, L, E):
    raw_grad = torch.zeros(raw_size, device=raw_arg.device, dtype=grad_sp.values().dtype if grad_sp._nnz() > 0 else torch.float)
    if raw_size == 0 or out_keys.numel() == 0:
        return raw_grad
    grad_sp = grad_sp.coalesce()
    grad_keys = grad_sp.indices()[0] * (L * E) + grad_sp.indices()[1] * E + grad_sp.indices()[2]
    grad_vals = grad_sp.values()
    pos = torch.searchsorted(out_keys, grad_keys)
    valid = (pos < out_keys.numel()) & (out_keys[pos] == grad_keys)
    if valid.any():
        chosen_raw = raw_arg[pos[valid]]
        chosen_valid = chosen_raw >= 0
        if chosen_valid.any():
            raw_grad[chosen_raw[chosen_valid]] = grad_vals[valid][chosen_valid]
    return raw_grad


def _align_sparse_grad_to_keys(grad_sp, out_keys, raw_b, raw_l, raw_e, L, E, dtype):
    if raw_b.numel() == 0:
        return torch.empty(0, device=raw_b.device, dtype=dtype)
    raw_keys = raw_b * (L * E) + raw_l * E + raw_e
    grad_sp = grad_sp.coalesce()
    grad_keys = grad_sp.indices()[0] * (L * E) + grad_sp.indices()[1] * E + grad_sp.indices()[2]
    grad_vals = grad_sp.values()
    out_grad = torch.zeros(out_keys.shape[0], device=raw_b.device, dtype=dtype)
    if grad_keys.numel() > 0 and out_keys.numel() > 0:
        pos = torch.searchsorted(out_keys, grad_keys)
        valid_pos = pos < out_keys.numel()
        safe_pos = pos.clamp(max=max(out_keys.numel() - 1, 0))
        valid = valid_pos & (out_keys[safe_pos] == grad_keys)
        out_grad[safe_pos[valid]] = grad_vals[valid].to(dtype)
    raw_pos = torch.searchsorted(out_keys, raw_keys)
    raw_grad = torch.zeros(raw_keys.shape[0], device=raw_b.device, dtype=dtype)
    valid_raw_pos = raw_pos < out_keys.numel()
    safe_raw_pos = raw_pos.clamp(max=max(out_keys.numel() - 1, 0))
    valid = valid_raw_pos & (out_keys[safe_raw_pos] == raw_keys)
    raw_grad[valid] = out_grad[safe_raw_pos[valid]]
    return raw_grad


def _reduce_sparse3d_maxgroup_from_raw(raw_b, raw_l, raw_e, raw_v, raw_entry, raw_rel, raw_mask, B, L, E, r_size):
    device = raw_v.device
    if raw_v.numel() == 0:
        empty_i = torch.empty(0, device=device, dtype=torch.long)
        empty_s = torch.empty(0, device=device, dtype=torch.short)
        empty_v = torch.empty(0, device=device, dtype=raw_v.dtype)
        idx = torch.stack([empty_i, empty_i, empty_i], dim=0)
        sp = torch.sparse_coo_tensor(idx, empty_v, [B, L, E]).coalesce()
        return sp, empty_i, empty_i, empty_i, empty_i, empty_i, empty_s, empty_v

    key4 = ((raw_b * L + raw_l) * E + raw_e) * r_size + raw_rel.long()
    sort_idx = torch.argsort(key4)
    sorted_key4 = key4[sort_idx]
    sorted_values = raw_v[sort_idx]

    unique_key4, inverse4 = torch.unique_consecutive(sorted_key4, return_inverse=True)
    red_values = torch.full((unique_key4.shape[0],), float("-inf"), device=device, dtype=raw_v.dtype)
    red_values.scatter_reduce_(0, inverse4, sorted_values, reduce="amax", include_self=True)

    sorted_pos = torch.arange(sorted_values.shape[0], device=device, dtype=torch.long)
    sentinel = sorted_values.shape[0]
    matched = sorted_values == red_values[inverse4]
    candidate = torch.where(matched, sorted_pos, torch.full_like(sorted_pos, sentinel))
    arg_sorted_pos = torch.full((unique_key4.shape[0],), sentinel, device=device, dtype=torch.long)
    arg_sorted_pos.scatter_reduce_(0, inverse4, candidate, reduce="amin", include_self=True)
    arg_raw = sort_idx[arg_sorted_pos.clamp(max=max(sorted_values.shape[0] - 1, 0))]
    arg_raw = torch.where(arg_sorted_pos < sentinel, arg_raw, torch.full_like(arg_raw, -1))

    red_entry = raw_entry[arg_raw]
    red_rel = raw_rel[arg_raw]
    red_mask = raw_mask[arg_raw]

    key3 = unique_key4 // r_size
    unique_key3, inverse3 = torch.unique_consecutive(key3, return_inverse=True)
    out_values = torch.zeros(unique_key3.shape[0], device=device, dtype=raw_v.dtype)
    out_values.scatter_add_(0, inverse3, red_values)

    out_b = unique_key3 // (L * E)
    rem = unique_key3 % (L * E)
    out_l = rem // E
    out_e = rem % E
    red_b = key3 // (L * E)
    red_rem = key3 % (L * E)
    red_l = red_rem // E
    red_e = red_rem % E
    sp = torch.sparse_coo_tensor(torch.stack([out_b, out_l, out_e], dim=0), out_values, [B, L, E]).coalesce()
    return sp, unique_key3, red_b, red_l, red_e, red_entry, red_rel, red_mask


def _gather_sparse3d_grad_at_entities(grad_sp, b, l, e, L, E):
    if b.numel() == 0:
        return torch.empty(0, device=b.device, dtype=torch.float)
    grad_sp = grad_sp.coalesce()
    grad_keys = grad_sp.indices()[0] * (L * E) + grad_sp.indices()[1] * E + grad_sp.indices()[2]
    grad_vals = grad_sp.values()
    out = torch.zeros(b.shape[0], device=b.device, dtype=grad_vals.dtype)
    if grad_keys.numel() == 0:
        return out
    chunk_size = 1_000_000
    for start in range(0, int(b.numel()), chunk_size):
        end = min(start + chunk_size, int(b.numel()))
        query = b[start:end] * (L * E) + l[start:end] * E + e[start:end]
        pos = torch.searchsorted(grad_keys, query)
        valid_pos = pos < grad_keys.numel()
        safe_pos = pos.clamp(max=max(grad_keys.numel() - 1, 0))
        valid = valid_pos & (grad_keys[safe_pos] == query)
        if bool(valid.any().item()):
            out[start:end][valid] = grad_vals[safe_pos[valid]]
    return out


def _finalize_sparse3d_maxgroup_outputs(
    sp_batch, sp_level, sp_entity, sp_value, w,
    ori_b, ori_l, ori_e, ori_v, ori_meta_entry, ori_meta_rel, ori_meta_mask,
    inv_b, inv_l, inv_e, inv_v, inv_meta_entry, inv_meta_rel, inv_meta_mask,
    ind_b, ind_l, ind_e, ind_v,
    B, L, E, r_size
):
    out_ori, ori_keys, ori_red_b, ori_red_l, ori_red_e, ori_red_entry, ori_red_rel, ori_red_mask = \
        _reduce_sparse3d_maxgroup_from_raw(
            ori_b, ori_l, ori_e, ori_v, ori_meta_entry, ori_meta_rel, ori_meta_mask, B, L, E, r_size
        )
    out_inv, inv_keys, inv_red_b, inv_red_l, inv_red_e, inv_red_entry, inv_red_rel, inv_red_mask = \
        _reduce_sparse3d_maxgroup_from_raw(
            inv_b, inv_l, inv_e, inv_v, inv_meta_entry, inv_meta_rel, inv_meta_mask, B, L, E, r_size
        )
    out_ind = torch.sparse_coo_tensor(torch.stack([ind_b, ind_l, ind_e], dim=0), ind_v, [B, L, E]).coalesce()

    saved = (
        sp_batch, sp_level, sp_entity, sp_value.float(), w.float(),
        ori_keys, ori_red_b, ori_red_l, ori_red_e, ori_red_entry, ori_red_rel, ori_red_mask,
        inv_keys, inv_red_b, inv_red_l, inv_red_e, inv_red_entry, inv_red_rel, inv_red_mask,
    )
    return out_ind, out_ori, out_inv, saved


def _backward_sparse3d_maxgroup_reduced(
    grad_ind, grad_ori, grad_inv,
    saved_tensors, L, E, r_size, wot_i
):
    (sp_batch, sp_level, sp_entity, sp_value, w,
     ori_keys, ori_red_b, ori_red_l, ori_red_e, ori_red_entry, ori_red_rel, ori_red_mask,
     inv_keys, inv_red_b, inv_red_l, inv_red_e, inv_red_entry, inv_red_rel, inv_red_mask) = saved_tensors

    n = w.size(2)
    grad_sp_value = torch.zeros_like(sp_value)
    grad_w = torch.zeros_like(w)

    ori_red_grad = _align_sparse_grad_to_keys(
        grad_ori, ori_keys, ori_red_b, ori_red_l, ori_red_e, L, E, sp_value.dtype
    )
    inv_red_grad = _align_sparse_grad_to_keys(
        grad_inv, inv_keys, inv_red_b, inv_red_l, inv_red_e, L, E, sp_value.dtype
    )

    if ori_red_grad.numel() > 0:
        ori_entry = ori_red_entry.long()
        ori_b = sp_batch[ori_entry]
        ori_l = sp_level[ori_entry]
        ori_coeff = w[ori_b, ori_l, ori_red_rel.long()]
        grad_sp_value.index_add_(0, ori_entry, ori_red_grad * ori_red_mask * ori_coeff)
        ori_w_index = ((ori_b * L + ori_l) * n + ori_red_rel.long())
        grad_w.view(-1).index_add_(0, ori_w_index, ori_red_grad * ori_red_mask * sp_value[ori_entry])

    if inv_red_grad.numel() > 0:
        inv_entry = inv_red_entry.long()
        inv_b = sp_batch[inv_entry]
        inv_l = sp_level[inv_entry]
        inv_rel = inv_red_rel.long() + r_size
        inv_coeff = w[inv_b, inv_l, inv_rel]
        grad_sp_value.index_add_(0, inv_entry, inv_red_grad * inv_red_mask * inv_coeff)
        inv_w_index = ((inv_b * L + inv_l) * n + inv_rel)
        grad_w.view(-1).index_add_(0, inv_w_index, inv_red_grad * inv_red_mask * sp_value[inv_entry])

    if not wot_i and grad_ind is not None:
        grad_ind_at_entry = _gather_sparse3d_grad_at_entities(grad_ind, sp_batch, sp_level, sp_entity, L, E)
        id_coeff = w[sp_batch, sp_level, n - 1]
        grad_sp_value += grad_ind_at_entry * id_coeff
        id_w_index = ((sp_batch * L + sp_level) * n + (n - 1))
        grad_w.view(-1).index_add_(0, id_w_index, grad_ind_at_entry * sp_value)

    return grad_sp_value, grad_w


def _compute_active_nodes(A):
    pass                                                                 
                                                                     
    non_zero = torch.nonzero(A.sum(1))
    if non_zero.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=A.device)
    return torch.unique(non_zero[:, 1])


def _ind2ptr_cpu(index, size):
    pass                                                              
    num = torch.zeros(size, dtype=torch.long)
    num.scatter_add_(0, index, torch.ones(index.size(0), dtype=torch.long))
    ptr = num.cumsum(0) - num
    return torch.cat([ptr, torch.tensor([index.size(0)], dtype=torch.long)])


def _build_group_metadata(sorted_rows, sorted_rels, num_nodes):
    if sorted_rows.numel() == 0:
        empty_i32 = torch.empty(0, dtype=torch.int32)
        empty_i16 = torch.empty(0, dtype=torch.int16)
        return _ind2ptr_cpu(torch.empty(0, dtype=torch.long), num_nodes).to(torch.int32), empty_i16, empty_i32, empty_i32

    starts = torch.ones(sorted_rows.size(0), dtype=torch.bool)
    starts[1:] = (sorted_rows[1:] != sorted_rows[:-1]) | (sorted_rels[1:] != sorted_rels[:-1])
    group_edge_start = starts.nonzero(as_tuple=True)[0]
    group_rows = sorted_rows[group_edge_start]
    group_rel = sorted_rels[group_edge_start].to(torch.int16)
    next_start = torch.cat([group_edge_start[1:], torch.tensor([sorted_rows.size(0)], dtype=torch.long)])
    group_edge_count = (next_start - group_edge_start).to(torch.int32)
    row_group_ptr = _ind2ptr_cpu(group_rows, num_nodes).to(torch.int32)
    return row_group_ptr, group_rel, group_edge_start.to(torch.int32), group_edge_count


def build_csr_structure(row_indices, col_indices, r_indices, num_nodes):
    pass                                                               
                                                                      
                                                                           
       
    device = row_indices.device

                             
    row_cpu = row_indices.long().cpu()
    col_cpu = col_indices.long().cpu()
    r_cpu   = r_indices.long().cpu()

    rel_base = int(r_cpu.max().item()) + 1 if r_cpu.numel() > 0 else 1

                                          
    order_ori       = (row_cpu * rel_base + r_cpu).argsort()
    ori_sorted_rows = row_cpu[order_ori]
    ori_col_ind_cpu = col_cpu[order_ori].to(torch.int32)
    ori_r_ind_cpu   = r_cpu[order_ori].to(torch.int16)
    ori_row_ptr_cpu = _ind2ptr_cpu(ori_sorted_rows, num_nodes)
    ori_row_group_ptr_cpu, ori_group_rel_cpu, ori_group_edge_start_cpu, ori_group_edge_count_cpu = \
        _build_group_metadata(ori_sorted_rows, ori_r_ind_cpu.to(torch.long), num_nodes)

                                          
    order_inv       = (col_cpu * rel_base + r_cpu).argsort()
    inv_sorted_cols = col_cpu[order_inv]
    inv_col_ind_cpu = row_cpu[order_inv].to(torch.int32)           
    inv_r_ind_cpu   = r_cpu[order_inv].to(torch.int16)
    inv_row_ptr_cpu = _ind2ptr_cpu(inv_sorted_cols, num_nodes)
    inv_row_group_ptr_cpu, inv_group_rel_cpu, inv_group_edge_start_cpu, inv_group_edge_count_cpu = \
        _build_group_metadata(inv_sorted_cols, inv_r_ind_cpu.to(torch.long), num_nodes)

    base = (ori_row_ptr_cpu.to(device=device, dtype=torch.int32), ori_col_ind_cpu.to(device),
            ori_r_ind_cpu.to(device),   order_ori.to(device=device, dtype=torch.int32),
            inv_row_ptr_cpu.to(device=device, dtype=torch.int32), inv_col_ind_cpu.to(device),
            inv_r_ind_cpu.to(device),   order_inv.to(device=device, dtype=torch.int32),
            ori_row_group_ptr_cpu.to(device), ori_group_rel_cpu.to(device),
            ori_group_edge_start_cpu.to(device), ori_group_edge_count_cpu.to(device),
            inv_row_group_ptr_cpu.to(device), inv_group_rel_cpu.to(device),
            inv_group_edge_start_cpu.to(device), inv_group_edge_count_cpu.to(device))
    return base


def build_smgroup_metadata(row_indices, col_indices, r_indices, num_nodes):
    row_cpu = row_indices.long().cpu()
    col_cpu = col_indices.long().cpu()
    r_cpu = r_indices.long().cpu()
    rel_base = int(r_cpu.max().item()) + 1 if r_cpu.numel() > 0 else 1

    def build_side(src_cpu, dst_cpu, rel_cpu):
        num_edges = src_cpu.numel()
        empty_i32 = torch.empty(0, dtype=torch.int32)
        empty_i16 = torch.empty(0, dtype=torch.int16)
        if num_edges == 0:
            return (
                empty_i32, empty_i32, empty_i32, empty_i16, empty_i32, empty_i32,
                torch.zeros(num_nodes + 1, dtype=torch.int32), empty_i32, empty_i32, empty_i16, empty_i32, empty_i32,
                empty_i32, empty_i32,
                torch.zeros(num_nodes + 1, dtype=torch.int32), empty_i32, empty_i32, empty_i16,
                empty_i32, empty_i32, empty_i32
            )

        row_order = (src_cpu * rel_base + rel_cpu).argsort()
        row_sorted_src = src_cpu[row_order]
        row_sorted_rel = rel_cpu[row_order].to(torch.int16)
        row_starts = torch.ones(num_edges, dtype=torch.bool)
        row_starts[1:] = (row_sorted_src[1:] != row_sorted_src[:-1]) | (row_sorted_rel[1:] != row_sorted_rel[:-1])
        row_group_edge_start = row_starts.nonzero(as_tuple=True)[0].to(torch.int32)
        row_next = torch.cat([row_group_edge_start[1:].to(torch.long), torch.tensor([num_edges], dtype=torch.long)])
        row_group_edge_count = (row_next - row_group_edge_start.to(torch.long)).to(torch.int32)
        row_group_rel = row_sorted_rel[row_group_edge_start.long()]
        row_group_ptr = _ind2ptr_cpu(row_sorted_src[row_group_edge_start.long()], num_nodes).to(torch.int32)
        row_group_index_sorted = row_starts.to(torch.int32).cumsum(0) - 1
        row_group_index_orig = torch.empty(num_edges, dtype=torch.int32)
        row_group_index_orig[row_order] = row_group_index_sorted.to(torch.int32)
        row_offset_sorted = torch.arange(num_edges, dtype=torch.long) - row_group_edge_start[row_group_index_sorted.long()].to(torch.long)
        row_offset_orig = torch.empty(num_edges, dtype=torch.int32)
        row_offset_orig[row_order] = row_offset_sorted.to(torch.int32)

        order = (dst_cpu * rel_base + rel_cpu).argsort()
        sorted_src = src_cpu[order].to(torch.int32)
        sorted_dst = dst_cpu[order]
        sorted_rel = rel_cpu[order].to(torch.int16)
        starts = torch.ones(num_edges, dtype=torch.bool)
        starts[1:] = (sorted_dst[1:] != sorted_dst[:-1]) | (sorted_rel[1:] != sorted_rel[:-1])
        group_edge_start = starts.nonzero(as_tuple=True)[0].to(torch.int32)
        next_start = torch.cat([group_edge_start[1:].to(torch.long), torch.tensor([num_edges], dtype=torch.long)])
        group_edge_count = (next_start - group_edge_start.to(torch.long)).to(torch.int32)
        group_dst = sorted_dst[group_edge_start.long()].to(torch.int32)
        group_rel = sorted_rel[group_edge_start.long()]

        global_group_keys = group_dst.to(torch.long) * rel_base + group_rel.to(torch.long)
        local_key_base = num_nodes * rel_base
        local_order = (src_cpu * local_key_base + dst_cpu * rel_base + rel_cpu).argsort()
        local_sorted_src = src_cpu[local_order]
        local_sorted_dst = dst_cpu[local_order]
        local_sorted_rel = rel_cpu[local_order].to(torch.int16)
        local_starts = torch.ones(num_edges, dtype=torch.bool)
        local_starts[1:] = (
            (local_sorted_src[1:] != local_sorted_src[:-1]) |
            (local_sorted_dst[1:] != local_sorted_dst[:-1]) |
            (local_sorted_rel[1:] != local_sorted_rel[:-1])
        )
        local_group_edge_start = local_starts.nonzero(as_tuple=True)[0].to(torch.int32)
        local_next = torch.cat([local_group_edge_start[1:].to(torch.long), torch.tensor([num_edges], dtype=torch.long)])
        local_group_edge_count = (local_next - local_group_edge_start.to(torch.long)).to(torch.int32)
        local_group_src = local_sorted_src[local_group_edge_start.long()].to(torch.int32)
        local_group_rel = local_sorted_rel[local_group_edge_start.long()]
        local_group_dst = local_sorted_dst[local_group_edge_start.long()].to(torch.int32)
        local_group_keys = local_group_dst.to(torch.long) * rel_base + local_group_rel.to(torch.long)
        local_group_global = torch.searchsorted(global_group_keys, local_group_keys).to(torch.int32)
        src_group_ptr = _ind2ptr_cpu(local_group_src.to(torch.long), num_nodes).to(torch.int32)
        local_group_starts = local_starts.to(torch.int32).cumsum(0) - 1

        return (
            sorted_src, group_edge_start, group_edge_count, group_rel, group_dst, order.to(torch.int32),
            src_group_ptr, local_group_edge_start, local_group_edge_count, local_group_rel, local_group_dst,
            local_order.to(torch.int32), local_group_src, local_group_global,
            local_group_starts.to(torch.int32), row_group_index_orig[local_order].to(torch.int32), row_offset_orig[local_order].to(torch.int32)
        )

    ori = build_side(row_cpu, col_cpu, r_cpu)
    inv = build_side(col_cpu, row_cpu, r_cpu)
    return ori[:14] + inv[:14] + (ori[14], ori[15], ori[16], inv[14], inv[15], inv[16])


def apply_mask(csr_struct, mask_values, weight=None):
    pass                                                                  
                                                                   
       
    (ori_row_ptr, ori_col_ind, ori_r_ind, order_ori,
     inv_row_ptr, inv_col_ind, inv_r_ind, order_inv, *_) = csr_struct

    mask_values = mask_values.float()
    if weight is not None:
        ew = score_function(weight).squeeze(-1)
        mask_with_weight = mask_values * ew
    else:
        mask_with_weight = mask_values

    ori_mask = mask_with_weight[order_ori]
    inv_mask = mask_with_weight[order_inv]

    return (ori_row_ptr, ori_col_ind, ori_r_ind, ori_mask,
            inv_row_ptr, inv_col_ind, inv_r_ind, inv_mask)


def build_csr(row_indices, col_indices, r_indices, mask_values, num_nodes, weight=None):
    pass                                                                   
       
    struct = build_csr_structure(row_indices, col_indices, r_indices, num_nodes)
    return apply_mask(struct, mask_values, weight=weight)


class FastLogFunction(autograd.Function):
    @staticmethod
    def forward(ctx, A, w, active_nodes,
                ori_col_ind, ori_mask,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_col_ind, inv_mask,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                r_size, wot_i, topk_edges):
        module = get_module()

        B, L, E = A.shape
        A_flat = A.reshape(B * L, E)
        w_flat = w.reshape(B * L, -1)

        A_f = A_flat if A_flat.is_floating_point() else A_flat.float()
        w_f = w_flat if w_flat.is_floating_point() else w_flat.float()

        if A.is_cuda:
            out_ind, out_ori, out_inv = module.fastlog_forward_cuda(
                A_f, w_f, active_nodes,
                torch.empty(0, device=A.device, dtype=torch.int32),
                ori_col_ind, torch.empty(0, device=A.device, dtype=torch.int16),
                ori_mask,
                torch.empty(0, device=A.device, dtype=torch.int32),
                inv_col_ind, torch.empty(0, device=A.device, dtype=torch.int16),
                inv_mask,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                r_size, wot_i, topk_edges > 0, topk_edges, 0
            )
        else:
            out_ind, out_ori, out_inv = module.fastlog_forward_cpu(
                A_f, w_f, active_nodes,
                ori_col_ind, ori_mask,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_col_ind, inv_mask,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                r_size, wot_i, topk_edges
            )

        out_ind = out_ind.reshape(B, L, E)
        out_ori = out_ori.reshape(B, L, E)
        out_inv = out_inv.reshape(B, L, E)

        ctx.save_for_backward(A, w, active_nodes,
                              ori_col_ind, ori_mask,
                              ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                              inv_col_ind, inv_mask,
                              inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count)
        ctx.r_size = r_size
        ctx.wot_i = wot_i
        ctx.topk_edges = topk_edges
        return out_ind, out_ori, out_inv

    @staticmethod
    def backward(ctx, grad_ind, grad_ori, grad_inv):
        (A, w, active_nodes,
         ori_col_ind, ori_mask,
         ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
         inv_col_ind, inv_mask,
         inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count) = ctx.saved_tensors
        module = get_module()

        B, L, E = A.shape
        BL = B * L
        A_f = A.reshape(BL, E).contiguous().float()
        w_f = w.reshape(BL, -1).contiguous().float()
        grad_ind_flat = grad_ind.reshape(BL, E).contiguous()
        grad_ori_flat = grad_ori.reshape(BL, E).contiguous()
        grad_inv_flat = grad_inv.reshape(BL, E).contiguous()

        if A.is_cuda:
            grad_A, grad_w = module.fastlog_backward_cuda(
                grad_ind_flat, grad_ori_flat, grad_inv_flat,
                torch.zeros(BL, E, device=A.device, dtype=A.dtype),
                torch.zeros(BL, E, device=A.device, dtype=A.dtype),
                A_f, w_f, active_nodes,
                torch.empty(0, device=A.device, dtype=torch.int32),
                ori_col_ind, torch.empty(0, device=A.device, dtype=torch.int16),
                ori_mask,
                torch.empty(0, device=A.device, dtype=torch.int32),
                inv_col_ind, torch.empty(0, device=A.device, dtype=torch.int16),
                inv_mask,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                ctx.r_size, ctx.wot_i, ctx.topk_edges > 0, ctx.topk_edges, 0
            )
        else:
            grad_A, grad_w = module.fastlog_backward_cpu(
                grad_ind_flat, grad_ori_flat, grad_inv_flat,
                A_f, w_f, active_nodes,
                ori_col_ind, ori_mask,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_col_ind, inv_mask,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                ctx.r_size, ctx.wot_i, ctx.topk_edges
            )

        grad_A = grad_A.reshape(B, L, E)
        grad_w = grad_w.reshape(B, L, -1)

        return (grad_A, grad_w, None,
                None, None,
                None, None, None, None,
                None, None,
                None, None, None, None,
                None, None, None)


def vectorized_operation(A, B, target_size, r_size, is_max=False, topk_pruning=100000,
                         weight=None, use_topk=False, wot_i=False):
    pass                                                       
                                                                                        
       
    row_indices, col_indices, r_indices, mask_values, w = B

                                                        
    csr_struct = build_csr_structure(row_indices, col_indices, r_indices, target_size)

    return vectorized_operation_csr(
        A, w, csr_struct, mask_values, r_size,
        weight=weight, wot_i=wot_i,
        use_topk=use_topk, topk_nodes=topk_pruning, topk_edges=0
    )


def vectorized_operation_csr(A, w, csr_struct, mask_values, r_size,
                              weight=None, wot_i=False,
                              use_topk=False, topk_nodes=100000, topk_edges=0):
    pass                                                                      
                                                        

                                                          
                                                                
       
    (ori_row_ptr, ori_col_ind, ori_r_ind, order_ori,
     inv_row_ptr, inv_col_ind, inv_r_ind, order_inv,
     ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
     inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count) = csr_struct

                                                                       
    mask_f = mask_values.float()
    if weight is not None:
        ew = score_function(weight).squeeze(-1)
        mask_f = mask_f * ew
    ori_mask = mask_f[order_ori]
    inv_mask = mask_f[order_inv]

                                  
    active_nodes = _compute_active_nodes(A)
    if use_topk and active_nodes.size(0) > topk_nodes:
        B, L, E = A.shape
        A_flat = A.reshape(B * L, E)
        scores = A_flat[:, active_nodes].sum(dim=0)
        _, topk_idx = torch.topk(scores, k=topk_nodes)
        active_nodes = active_nodes[topk_idx].sort()[0]

                                                                          
    effective_topk = topk_edges if (use_topk and topk_edges > 0) else (1 << 60)

    return FastLogFunction.apply(
        A, w, active_nodes,
        ori_col_ind, ori_mask,
        ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
        inv_col_ind, inv_mask,
        inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
        r_size, wot_i, effective_topk
    )


class FastLogFunctionV2(autograd.Function):
    pass                                                        
                                                                     
                                                                              
       
    @staticmethod
    def forward(ctx, A, w, ori_row_ptr, ori_col_ind, ori_r_ind, ori_order,
                inv_row_ptr, inv_col_ind, inv_r_ind, inv_order,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                mask_values, weight, r_size, wot_i,
                use_topk, topk_nodes, topk_edges, agg_mode):
        module = get_module()
        ori_row_ptr = ori_row_ptr.int()
        ori_col_ind = ori_col_ind.int()
        ori_r_ind = ori_r_ind.short()
        ori_order = ori_order.int()
        inv_row_ptr = inv_row_ptr.int()
        inv_col_ind = inv_col_ind.int()
        inv_r_ind = inv_r_ind.short()
        inv_order = inv_order.int()

        B, L, E = A.shape
        BL = B * L
        A_flat = A.reshape(BL, E).contiguous().float()
        w_flat = w.reshape(BL, -1).contiguous().float()

                                           
        active_nodes = module.compute_active_nodes_cuda(A_flat)

                                           
        if use_topk and active_nodes.size(0) > topk_nodes:
            scores = A_flat[:, active_nodes].sum(dim=0)                
            _, topk_idx = torch.topk(scores, k=topk_nodes)
            active_nodes = active_nodes[topk_idx].sort()[0]

        mask_f = mask_values
        w_tensor = weight.float().squeeze(-1) if weight is not None else torch.empty(0, device=A.device)
        ori_mask = module.apply_mask_cuda(mask_f, ori_order, w_tensor)
        inv_mask = module.apply_mask_cuda(mask_f, inv_order, w_tensor)

                                      
        out_ind, out_ori, out_inv = module.fastlog_forward_cuda(
            A_flat, w_flat, active_nodes,
            ori_row_ptr, ori_col_ind, ori_r_ind, ori_mask,
            inv_row_ptr, inv_col_ind, inv_r_ind, inv_mask,
            ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
            inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
            r_size, wot_i, use_topk, topk_edges, agg_mode
        )

        ctx.save_for_backward(A, w, active_nodes, out_ori.reshape(B, L, E), out_inv.reshape(B, L, E),
                              ori_row_ptr, ori_col_ind, ori_r_ind, ori_mask,
                              inv_row_ptr, inv_col_ind, inv_r_ind, inv_mask,
                              ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                              inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count)
        ctx.r_size = r_size
        ctx.wot_i = wot_i
        ctx.use_topk = use_topk
        ctx.topk_edges = topk_edges
        ctx.agg_mode = agg_mode

        return (out_ind.reshape(B, L, E),
                out_ori.reshape(B, L, E),
                out_inv.reshape(B, L, E))

    @staticmethod
    def backward(ctx, grad_ind, grad_ori, grad_inv):
        (A, w, active_nodes, out_ori, out_inv,
         ori_row_ptr, ori_col_ind, ori_r_ind, ori_mask,
         inv_row_ptr, inv_col_ind, inv_r_ind, inv_mask,
         ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
         inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count) = ctx.saved_tensors
        module = get_module()

        B, L, E = A.shape
        BL = B * L
        A_flat = A.reshape(BL, E).contiguous().float()
        w_flat = w.reshape(BL, -1).contiguous().float()

        grad_A, grad_w = module.fastlog_backward_cuda(
            grad_ind.reshape(BL, E).contiguous(),
            grad_ori.reshape(BL, E).contiguous(),
            grad_inv.reshape(BL, E).contiguous(),
            out_ori.reshape(BL, E).contiguous(),
            out_inv.reshape(BL, E).contiguous(),
            A_flat, w_flat, active_nodes,
            ori_row_ptr, ori_col_ind, ori_r_ind, ori_mask,
            inv_row_ptr, inv_col_ind, inv_r_ind, inv_mask,
            ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
            inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
            ctx.r_size, ctx.wot_i, ctx.use_topk, ctx.topk_edges, ctx.agg_mode
        )

        grad_A = grad_A.reshape(B, L, E)
        grad_w = grad_w.reshape(B, L, -1)

        return (grad_A, grad_w,
                None, None, None, None,
                None, None, None, None,
                None, None, None, None,
                None, None, None, None,
                None, None, None, None,
                None, None, None, None)


class FastLogFunctionV2MaxExact(autograd.Function):
    @staticmethod
    def forward(ctx, A, w, ori_row_ptr, ori_col_ind, ori_r_ind, ori_order,
                inv_row_ptr, inv_col_ind, inv_r_ind, inv_order,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                mask_values, weight, r_size, wot_i,
                use_topk, topk_nodes, topk_edges):
        module = get_module()
        ori_row_ptr = ori_row_ptr.int()
        ori_col_ind = ori_col_ind.int()
        ori_r_ind = ori_r_ind.short()
        ori_order = ori_order.int()
        inv_row_ptr = inv_row_ptr.int()
        inv_col_ind = inv_col_ind.int()
        inv_r_ind = inv_r_ind.short()
        inv_order = inv_order.int()

        B, L, E = A.shape
        BL = B * L
        A_flat = A.reshape(BL, E).contiguous().float()
        w_flat = w.reshape(BL, -1).contiguous().float()

        active_nodes = module.compute_active_nodes_cuda(A_flat)
        if use_topk and active_nodes.size(0) > topk_nodes:
            scores = A_flat[:, active_nodes].sum(dim=0)
            _, topk_idx = torch.topk(scores, k=topk_nodes)
            active_nodes = active_nodes[topk_idx].sort()[0]

        mask_f = mask_values
        w_tensor = weight.float().squeeze(-1) if weight is not None else torch.empty(0, device=A.device)
        ori_mask = module.apply_mask_cuda(mask_f, ori_order, w_tensor)
        inv_mask = module.apply_mask_cuda(mask_f, inv_order, w_tensor)

        out_ind, out_ori, out_inv, ori_arg, inv_arg = module.fastlog_forward_max_cuda(
            A_flat, w_flat, active_nodes,
            ori_row_ptr, ori_col_ind, ori_r_ind, ori_mask,
            inv_row_ptr, inv_col_ind, inv_r_ind, inv_mask,
            ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
            inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
            r_size, wot_i, use_topk, topk_edges
        )

        ctx.save_for_backward(
            A, w, active_nodes, ori_arg, inv_arg,
            ori_row_ptr, ori_col_ind, ori_r_ind, ori_mask,
            inv_row_ptr, inv_col_ind, inv_r_ind, inv_mask,
            ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
            inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count
        )
        ctx.r_size = r_size
        ctx.wot_i = wot_i
        ctx.use_topk = use_topk
        ctx.topk_edges = topk_edges

        return (out_ind.reshape(B, L, E),
                out_ori.reshape(B, L, E),
                out_inv.reshape(B, L, E))

    @staticmethod
    def backward(ctx, grad_ind, grad_ori, grad_inv):
        (A, w, active_nodes, ori_arg, inv_arg,
         ori_row_ptr, ori_col_ind, ori_r_ind, ori_mask,
         inv_row_ptr, inv_col_ind, inv_r_ind, inv_mask,
         ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
         inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count) = ctx.saved_tensors
        module = get_module()

        B, L, E = A.shape
        BL = B * L
        A_flat = A.reshape(BL, E).contiguous().float()
        w_flat = w.reshape(BL, -1).contiguous().float()

        grad_A, grad_w = module.fastlog_backward_max_cuda(
            grad_ind.reshape(BL, E).contiguous(),
            grad_ori.reshape(BL, E).contiguous(),
            grad_inv.reshape(BL, E).contiguous(),
            ori_arg.contiguous(),
            inv_arg.contiguous(),
            A_flat, w_flat, active_nodes,
            ori_row_ptr, ori_col_ind, ori_r_ind, ori_mask,
            inv_row_ptr, inv_col_ind, inv_r_ind, inv_mask,
            ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
            inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
            ctx.r_size, ctx.wot_i, ctx.use_topk, ctx.topk_edges
        )

        grad_A = grad_A.reshape(B, L, E)
        grad_w = grad_w.reshape(B, L, -1)

        return (grad_A, grad_w,
                None, None, None, None,
                None, None, None, None,
                None, None, None, None,
                None, None, None, None,
                None, None, None, None,
                None, None, None)


class FastLogFunctionSparse3DTopK(autograd.Function):
    @staticmethod
    def forward(ctx, sp_batch, sp_level, sp_entity, sp_value, w,
                ori_row_ptr, ori_col_ind, ori_r_ind, ori_order,
                inv_row_ptr, inv_col_ind, inv_r_ind, inv_order,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                mask_values, weight, B, L, E, r_size, wot_i, topk_edges, agg_mode=0,
                edge_weight_is_score=False, edge_weight_scale=1.0):
        t_stage = time.perf_counter()
        module = get_module()
        _kernel_log("sparse3d_topk.get_module", t_stage)
        t_stage = time.perf_counter()
        ori_col_ind = ori_col_ind.int()
        ori_order = ori_order.int()
        inv_col_ind = inv_col_ind.int()
        inv_order = inv_order.int()
        mask_f = mask_values
        edge_weight = weight if weight is not None else torch.empty(0, device=sp_value.device)
        edge_weight_is_score = bool(edge_weight_is_score)
        edge_weight_scale = float(edge_weight_scale)
        _kernel_log("sparse3d_topk.prepare_inputs", t_stage)

        t_stage = time.perf_counter()
        (ori_b, ori_l, ori_e, ori_v,
         inv_b, inv_l, inv_e, inv_v,
         ind_b, ind_l, ind_e, ind_v,
         ori_meta_entry, ori_meta_rel, ori_meta_mask,
         inv_meta_entry, inv_meta_rel, inv_meta_mask,
         ori_offsets, inv_offsets,
         ori_meta_edge, inv_meta_edge) = module.fastlog_forward_sparse3d_topk_cuda(
            sp_batch, sp_level, sp_entity, sp_value, w,
            ori_col_ind, ori_order,
            ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
            inv_col_ind, inv_order,
            inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
            mask_f, edge_weight,
            B, L, E, r_size, wot_i, topk_edges, edge_weight_is_score, edge_weight_scale
        )
        _kernel_log("sparse3d_topk.cuda_forward", t_stage)
        if os.environ.get("FASTLOG_SPARSE_DEBUG_COUNTS", "") == "1":
            print(
                "Sparse3DTopKCounts nnz={} ori_total={} inv_total={} ind_total={}".format(
                    int(sp_value.numel()),
                    int(ori_v.numel()),
                    int(inv_v.numel()),
                    int(ind_v.numel()),
                ),
                flush=True,
            )

        t_stage = time.perf_counter()
        reduce_mode = "max" if agg_mode == 1 else "sum"
        if agg_mode == 1:
            ori_sp, ori_keys, ori_out_values, ori_raw_arg = _coalesce_sparse3d_max_with_arg(
                ori_b, ori_l, ori_e, ori_v, B, L, E, tie_break=ori_meta_edge
            )
            inv_sp, inv_keys, inv_out_values, inv_raw_arg = _coalesce_sparse3d_max_with_arg(
                inv_b, inv_l, inv_e, inv_v, B, L, E, tie_break=inv_meta_edge
            )
        else:
            ori_sp, ori_keys, ori_out_values = _coalesce_sparse3d_reduce(ori_b, ori_l, ori_e, ori_v, B, L, E, reduce=reduce_mode)
            inv_sp, inv_keys, inv_out_values = _coalesce_sparse3d_reduce(inv_b, inv_l, inv_e, inv_v, B, L, E, reduce=reduce_mode)
            ori_raw_arg = torch.empty(0, device=ori_v.device, dtype=torch.long)
            inv_raw_arg = torch.empty(0, device=inv_v.device, dtype=torch.long)
        ind_sp, ind_keys, ind_out_values = _coalesce_sparse3d_reduce(ind_b, ind_l, ind_e, ind_v, B, L, E, reduce="sum")
        _kernel_log("sparse3d_topk.python_reduce", t_stage)

        t_stage = time.perf_counter()
        needs_backward = any(bool(flag) for flag in ctx.needs_input_grad)
        if needs_backward:
            ctx.save_for_backward(
                sp_batch, sp_level, sp_entity, sp_value, w,
                ori_b, ori_l, ori_e, ori_v, ori_keys, ori_out_values, ori_raw_arg,
                ori_meta_entry, ori_meta_rel, ori_meta_mask, ori_offsets, ori_meta_edge,
                inv_b, inv_l, inv_e, inv_v, inv_keys, inv_out_values, inv_raw_arg,
                inv_meta_entry, inv_meta_rel, inv_meta_mask, inv_offsets, inv_meta_edge,
                edge_weight if weight is not None and bool(getattr(weight, "requires_grad", False)) else torch.empty(0, device=sp_value.device)
            )
            ctx.has_edge_weight = weight is not None and bool(getattr(weight, "requires_grad", False))
            ctx.edge_weight_is_score = edge_weight_is_score
            ctx.edge_weight_scale = edge_weight_scale
            ctx.B, ctx.L, ctx.E, ctx.r_size, ctx.wot_i, ctx.agg_mode = B, L, E, r_size, wot_i, agg_mode
        _kernel_log("sparse3d_topk.save_ctx", t_stage)
        return ori_sp, inv_sp, ind_sp

    @staticmethod
    def backward(ctx, grad_ori, grad_inv, grad_ind):
        (sp_batch, sp_level, sp_entity, sp_value, w,
         ori_b, ori_l, ori_e, ori_v, ori_keys, ori_out_values, ori_raw_arg,
         ori_meta_entry, ori_meta_rel, ori_meta_mask, ori_offsets, ori_meta_edge,
         inv_b, inv_l, inv_e, inv_v, inv_keys, inv_out_values, inv_raw_arg,
         inv_meta_entry, inv_meta_rel, inv_meta_mask, inv_offsets, inv_meta_edge,
         edge_weight) = ctx.saved_tensors
        module = get_module()

        def unpack_grad3(g):
            g = g.coalesce()
            return g.values(), g.indices()[0], g.indices()[1], g.indices()[2]

        gindv, gindb, gindl, ginde = unpack_grad3(grad_ind)
        if ctx.agg_mode == 1:
            grad_ori_raw = _align_sparse_grad_to_raw_argmax(
                grad_ori, ori_keys, ori_raw_arg, ori_v.shape[0], ctx.L, ctx.E
            )
            grad_inv_raw = _align_sparse_grad_to_raw_argmax(
                grad_inv, inv_keys, inv_raw_arg, inv_v.shape[0], ctx.L, ctx.E
            )
            grad_sp_value, grad_w = module.fastlog_backward_sparse3d_topk_aligned_cuda(
                grad_ori_raw, ori_meta_entry, ori_meta_rel, ori_meta_mask,
                grad_inv_raw, inv_meta_entry, inv_meta_rel, inv_meta_mask,
                gindv, gindb, gindl, ginde,
                sp_batch, sp_level, sp_entity, sp_value, w,
                ctx.B, ctx.L, ctx.E, ctx.r_size, ctx.wot_i
            )
        else:
            gov, gob, gol, goe = unpack_grad3(grad_ori)
            giv, gib, gil, gie = unpack_grad3(grad_inv)
            grad_sp_value, grad_w = module.fastlog_backward_sparse3d_topk_cuda(
                gov, gob, gol, goe,
                giv, gib, gil, gie,
                gindv, gindb, gindl, ginde,
                sp_batch, sp_level, sp_entity, sp_value, w,
                ori_b, ori_l, ori_e,
                ori_meta_entry, ori_meta_rel, ori_meta_mask, ori_offsets,
                inv_b, inv_l, inv_e,
                inv_meta_entry, inv_meta_rel, inv_meta_mask, inv_offsets,
                ctx.B, ctx.L, ctx.E, ctx.r_size, ctx.wot_i
            )
            grad_ori_raw = _gather_sparse3d_grad_at_entities(grad_ori, ori_b, ori_l, ori_e, ctx.L, ctx.E)
            grad_inv_raw = _gather_sparse3d_grad_at_entities(grad_inv, inv_b, inv_l, inv_e, ctx.L, ctx.E)

        grad_edge_weight = None
        if ctx.has_edge_weight:
            edge_weight_flat = edge_weight.view(-1)
            sparse_edge_grad = os.environ.get("FASTLOG_SPARSE_EDGE_GRAD", "0") == "1"
            grad_edge_flat = None if sparse_edge_grad else torch.zeros_like(edge_weight_flat)
            sparse_edges = []
            sparse_values = []

            def add_edge_weight_grad(raw_grad, meta_entry, meta_rel, meta_mask, meta_edge, rel_offset):
                if meta_edge.numel() == 0:
                    return
                chunk_size = 1_000_000
                for start in range(0, int(meta_edge.numel()), chunk_size):
                    end = min(start + chunk_size, int(meta_edge.numel()))
                    entry = meta_entry[start:end].long()
                    rel = meta_rel[start:end].long()
                    edge = meta_edge[start:end].long()
                    b = sp_batch[entry]
                    l = sp_level[entry]
                    rule_weight = w[b, l, rel_offset + rel]
                    if ctx.edge_weight_is_score:
                        edge_score = edge_weight_flat[edge]
                        edge_grad_factor = edge_score.clamp_min(1e-12).reciprocal()
                    else:
                        edge_score = score_function(edge_weight_flat[edge] * ctx.edge_weight_scale)
                        edge_grad_factor = ctx.edge_weight_scale * (1.0 - edge_score)
                    mask_score = meta_mask[start:end].to(dtype=raw_grad.dtype)
                    contrib = raw_grad[start:end] * sp_value[entry] * rule_weight * mask_score * edge_grad_factor
                    if sparse_edge_grad:
                        sparse_edges.append(edge)
                        sparse_values.append(contrib)
                    else:
                        grad_edge_flat.index_add_(0, edge, contrib)

            add_edge_weight_grad(grad_ori_raw, ori_meta_entry, ori_meta_rel, ori_meta_mask, ori_meta_edge, 0)
            add_edge_weight_grad(grad_inv_raw, inv_meta_entry, inv_meta_rel, inv_meta_mask, inv_meta_edge, ctx.r_size)
            if sparse_edge_grad:
                if sparse_edges:
                    edge_idx = torch.cat(sparse_edges, dim=0).long()
                    edge_val = torch.cat(sparse_values, dim=0)
                    if edge_weight.dim() == 2 and edge_weight.shape[1] == 1:
                        zeros = torch.zeros_like(edge_idx)
                        indices = torch.stack([edge_idx, zeros], dim=0)
                        grad_edge_weight = torch.sparse_coo_tensor(
                            indices,
                            edge_val,
                            edge_weight.shape,
                            device=edge_weight.device,
                        ).coalesce()
                    else:
                        grad_edge_weight = torch.sparse_coo_tensor(
                            edge_idx.view(1, -1),
                            edge_val,
                            edge_weight.shape,
                            device=edge_weight.device,
                        ).coalesce()
                else:
                    empty_idx = torch.empty((edge_weight.dim(), 0), dtype=torch.long, device=edge_weight.device)
                    empty_val = torch.empty((0,), dtype=edge_weight.dtype, device=edge_weight.device)
                    grad_edge_weight = torch.sparse_coo_tensor(
                        empty_idx,
                        empty_val,
                        edge_weight.shape,
                        device=edge_weight.device,
                    ).coalesce()
            else:
                grad_edge_weight = grad_edge_flat.view_as(edge_weight)
        return (
            None, None, None, grad_sp_value, grad_w,
            None, None, None, None,
            None, None, None, None,
            None, None, None, None,
            None, None, None, None,
            None, grad_edge_weight,
            None, None, None, None, None, None, None, None, None
        )


class FastLogFunctionSparse3DTopKMasked(autograd.Function):
    @staticmethod
    def forward(ctx, sp_batch, sp_level, sp_entity, sp_value, w,
                ori_row_ptr, ori_col_ind, ori_r_ind, ori_order,
                inv_row_ptr, inv_col_ind, inv_r_ind, inv_order,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                mask_values, weight, B, L, E, r_size, wot_i, topk_edges, agg_mode=0):
        t_stage = time.perf_counter()
        module = get_module()
        _kernel_log("sparse3d_topk_masked.get_module", t_stage)
        t_stage = time.perf_counter()
        ori_col_ind = ori_col_ind.int()
        ori_order = ori_order.int()
        inv_col_ind = inv_col_ind.int()
        inv_order = inv_order.int()
        mask_f = mask_values.float()
        w_tensor = weight.float().squeeze(-1) if weight is not None else torch.empty(0, device=sp_value.device)
        ori_mask = module.apply_mask_cuda(mask_f, ori_order, w_tensor)
        inv_mask = module.apply_mask_cuda(mask_f, inv_order, w_tensor)
        _kernel_log("sparse3d_topk_masked.prepare_masks", t_stage)

        t_stage = time.perf_counter()
        (ori_b, ori_l, ori_e, ori_v,
         inv_b, inv_l, inv_e, inv_v,
         ind_b, ind_l, ind_e, ind_v,
         ori_meta_entry, ori_meta_rel, ori_meta_mask,
         inv_meta_entry, inv_meta_rel, inv_meta_mask,
         ori_offsets, inv_offsets) = module.fastlog_forward_sparse3d_topk_masked_cuda(
            sp_batch, sp_level, sp_entity, sp_value, w,
            ori_col_ind, ori_mask,
            ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
            inv_col_ind, inv_mask,
            inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
            B, L, E, r_size, wot_i, topk_edges
        )
        _kernel_log("sparse3d_topk_masked.cuda_forward", t_stage)

        t_stage = time.perf_counter()
        reduce_mode = "max" if agg_mode == 1 else "sum"
        if agg_mode == 1:
            ori_sp, ori_keys, ori_out_values, ori_raw_arg = _coalesce_sparse3d_max_with_arg(
                ori_b, ori_l, ori_e, ori_v, B, L, E
            )
            inv_sp, inv_keys, inv_out_values, inv_raw_arg = _coalesce_sparse3d_max_with_arg(
                inv_b, inv_l, inv_e, inv_v, B, L, E
            )
        else:
            ori_sp, ori_keys, ori_out_values = _coalesce_sparse3d_reduce(ori_b, ori_l, ori_e, ori_v, B, L, E, reduce=reduce_mode)
            inv_sp, inv_keys, inv_out_values = _coalesce_sparse3d_reduce(inv_b, inv_l, inv_e, inv_v, B, L, E, reduce=reduce_mode)
            ori_raw_arg = torch.empty(0, device=ori_v.device, dtype=torch.long)
            inv_raw_arg = torch.empty(0, device=inv_v.device, dtype=torch.long)
        ind_sp, ind_keys, ind_out_values = _coalesce_sparse3d_reduce(ind_b, ind_l, ind_e, ind_v, B, L, E, reduce="sum")
        _kernel_log("sparse3d_topk_masked.python_reduce", t_stage)

        t_stage = time.perf_counter()
        ctx.save_for_backward(
            sp_batch, sp_level, sp_entity, sp_value, w,
            ori_b, ori_l, ori_e, ori_v, ori_keys, ori_out_values, ori_raw_arg,
            ori_meta_entry, ori_meta_rel, ori_meta_mask, ori_offsets,
            inv_b, inv_l, inv_e, inv_v, inv_keys, inv_out_values, inv_raw_arg,
            inv_meta_entry, inv_meta_rel, inv_meta_mask, inv_offsets
        )
        ctx.B, ctx.L, ctx.E, ctx.r_size, ctx.wot_i, ctx.agg_mode = B, L, E, r_size, wot_i, agg_mode
        _kernel_log("sparse3d_topk_masked.save_ctx", t_stage)
        return ori_sp, inv_sp, ind_sp

    @staticmethod
    def backward(ctx, grad_ori, grad_inv, grad_ind):
        (sp_batch, sp_level, sp_entity, sp_value, w,
         ori_b, ori_l, ori_e, ori_v, ori_keys, ori_out_values, ori_raw_arg,
         ori_meta_entry, ori_meta_rel, ori_meta_mask, ori_offsets,
         inv_b, inv_l, inv_e, inv_v, inv_keys, inv_out_values, inv_raw_arg,
         inv_meta_entry, inv_meta_rel, inv_meta_mask, inv_offsets) = ctx.saved_tensors
        module = get_module()

        def unpack_grad3(g):
            g = g.coalesce()
            return g.values(), g.indices()[0], g.indices()[1], g.indices()[2]

        gindv, gindb, gindl, ginde = unpack_grad3(grad_ind)
        if ctx.agg_mode == 1:
            grad_ori_raw = _align_sparse_grad_to_raw_argmax(
                grad_ori, ori_keys, ori_raw_arg, ori_v.shape[0], ctx.L, ctx.E
            )
            grad_inv_raw = _align_sparse_grad_to_raw_argmax(
                grad_inv, inv_keys, inv_raw_arg, inv_v.shape[0], ctx.L, ctx.E
            )
            grad_sp_value, grad_w = module.fastlog_backward_sparse3d_topk_aligned_cuda(
                grad_ori_raw, ori_meta_entry, ori_meta_rel, ori_meta_mask,
                grad_inv_raw, inv_meta_entry, inv_meta_rel, inv_meta_mask,
                gindv, gindb, gindl, ginde,
                sp_batch, sp_level, sp_entity, sp_value, w,
                ctx.B, ctx.L, ctx.E, ctx.r_size, ctx.wot_i
            )
        else:
            gov, gob, gol, goe = unpack_grad3(grad_ori)
            giv, gib, gil, gie = unpack_grad3(grad_inv)
            grad_sp_value, grad_w = module.fastlog_backward_sparse3d_topk_cuda(
                gov, gob, gol, goe,
                giv, gib, gil, gie,
                gindv, gindb, gindl, ginde,
                sp_batch, sp_level, sp_entity, sp_value, w,
                ori_b, ori_l, ori_e,
                ori_meta_entry, ori_meta_rel, ori_meta_mask, ori_offsets,
                inv_b, inv_l, inv_e,
                inv_meta_entry, inv_meta_rel, inv_meta_mask, inv_offsets,
                ctx.B, ctx.L, ctx.E, ctx.r_size, ctx.wot_i
            )
        return (
            None, None, None, grad_sp_value, grad_w,
            None, None, None, None,
            None, None, None, None,
            None, None, None, None,
            None, None, None, None,
            None, None,
            None, None, None, None, None, None, None
        )


class FastLogFunctionMaxGroupV2(autograd.Function):
    @staticmethod
    def forward(ctx, A, w,
                ori_sorted_src, ori_group_edge_start, ori_group_edge_count, ori_group_rel, ori_group_dst, ori_sorted_edge_index,
                ori_src_group_ptr, ori_local_group_edge_start, ori_local_group_edge_count, ori_local_group_rel, ori_local_group_dst, ori_local_order, ori_local_group_src, ori_local_group_global,
                inv_sorted_src, inv_group_edge_start, inv_group_edge_count, inv_group_rel, inv_group_dst, inv_sorted_edge_index,
                inv_src_group_ptr, inv_local_group_edge_start, inv_local_group_edge_count, inv_local_group_rel, inv_local_group_dst, inv_local_order, inv_local_group_src, inv_local_group_global,
                ori_row_group_ptr, ori_row_group_edge_start, ori_row_group_edge_count, ori_row_group_rel, ori_local_edge_group, ori_local_edge_row_group, ori_local_edge_row_offset,
                inv_row_group_ptr, inv_row_group_edge_start, inv_row_group_edge_count, inv_row_group_rel, inv_local_edge_group, inv_local_edge_row_group, inv_local_edge_row_offset,
                mask_values, weight, r_size, wot_i, use_topk, topk_nodes, topk_edges):
        module = get_module()
        B, L, E = A.shape
        BL = B * L
        A_flat = A.reshape(BL, E).contiguous().float()
        w_flat = w.reshape(BL, -1).contiguous().float()
        active_nodes = module.compute_active_nodes_cuda(A_flat)
        if use_topk and active_nodes.size(0) > topk_nodes:
            scores = A_flat[:, active_nodes].sum(dim=0)
            _, topk_idx = torch.topk(scores, k=topk_nodes)
            active_nodes = active_nodes[topk_idx].sort()[0]
        effective_use_topk = use_topk
        if effective_use_topk and active_nodes.numel() > 0 and topk_edges > 0:
            def _max_active_edges(row_group_ptr, row_group_edge_count):
                prefix = torch.cat([
                    torch.zeros(1, device=row_group_edge_count.device, dtype=torch.long),
                    row_group_edge_count.long().cumsum(0)
                ], dim=0)
                act = active_nodes.long()
                start = row_group_ptr[act].long()
                end = row_group_ptr[act + 1].long()
                return (prefix[end] - prefix[start]).max()

            ori_max_edges = _max_active_edges(ori_row_group_ptr, ori_row_group_edge_count)
            inv_max_edges = _max_active_edges(inv_row_group_ptr, inv_row_group_edge_count)
            if active_nodes.size(0) <= topk_nodes and topk_edges >= int(max(ori_max_edges.item(), inv_max_edges.item())):
                effective_use_topk = False
        mask_f = mask_values.float()
        w_tensor = weight.float().squeeze(-1) if weight is not None else torch.empty(0, device=A.device)
        if effective_use_topk:
            ori_mask = module.apply_mask_cuda(mask_f, ori_local_order.int(), w_tensor)
            inv_mask = module.apply_mask_cuda(mask_f, inv_local_order.int(), w_tensor)
        else:
            ori_mask = module.apply_mask_cuda(mask_f, ori_sorted_edge_index.int(), w_tensor)
            inv_mask = module.apply_mask_cuda(mask_f, inv_sorted_edge_index.int(), w_tensor)

        out_ind, out_ori, out_inv, ori_arg, inv_arg = module.fastlog_forward_maxgroup_cuda(
            A_flat, w_flat, active_nodes,
            ori_sorted_src.int(), ori_group_edge_start.int(), ori_group_edge_count.int(), ori_group_rel.short(), ori_group_dst.int(), ori_mask,
            ori_src_group_ptr.int(), ori_local_group_edge_start.int(), ori_local_group_edge_count.int(), ori_local_group_rel.short(), ori_local_group_dst.int(), ori_local_order.int(), ori_local_group_src.int(), ori_local_group_global.int(),
            ori_row_group_ptr.int(), ori_row_group_edge_start.int(), ori_row_group_edge_count.int(), ori_row_group_rel.short(), ori_local_edge_group.int(), ori_local_edge_row_group.int(), ori_local_edge_row_offset.int(),
            inv_sorted_src.int(), inv_group_edge_start.int(), inv_group_edge_count.int(), inv_group_rel.short(), inv_group_dst.int(), inv_mask,
            inv_src_group_ptr.int(), inv_local_group_edge_start.int(), inv_local_group_edge_count.int(), inv_local_group_rel.short(), inv_local_group_dst.int(), inv_local_order.int(), inv_local_group_src.int(), inv_local_group_global.int(),
            inv_row_group_ptr.int(), inv_row_group_edge_start.int(), inv_row_group_edge_count.int(), inv_row_group_rel.short(), inv_local_edge_group.int(), inv_local_edge_row_group.int(), inv_local_edge_row_offset.int(),
            r_size, wot_i, effective_use_topk, topk_edges
        )
        ctx.save_for_backward(
            A, w, active_nodes, ori_arg, inv_arg,
            ori_sorted_src.int(), ori_group_rel.short(), ori_group_dst.int(), ori_mask,
            ori_src_group_ptr.int(), ori_local_group_edge_start.int(), ori_local_group_edge_count.int(), ori_local_group_rel.short(), ori_local_group_dst.int(), ori_local_order.int(), ori_local_group_src.int(),
            ori_local_edge_group.int(),
            inv_sorted_src.int(), inv_group_rel.short(), inv_group_dst.int(), inv_mask,
            inv_src_group_ptr.int(), inv_local_group_edge_start.int(), inv_local_group_edge_count.int(), inv_local_group_rel.short(), inv_local_group_dst.int(), inv_local_order.int(), inv_local_group_src.int()
            ,inv_local_edge_group.int()
        )
        ctx.r_size = r_size
        ctx.wot_i = wot_i
        ctx.use_topk = effective_use_topk
        ctx.topk_edges = topk_edges
        return out_ind.reshape(B, L, E), out_ori.reshape(B, L, E), out_inv.reshape(B, L, E)

    @staticmethod
    def backward(ctx, grad_ind, grad_ori, grad_inv):
        (A, w, active_nodes, ori_arg, inv_arg,
         ori_sorted_src, ori_group_rel, ori_group_dst, ori_mask,
         ori_src_group_ptr, ori_local_group_edge_start, ori_local_group_edge_count, ori_local_group_rel, ori_local_group_dst, ori_local_order, ori_local_group_src, ori_local_edge_group,
         inv_sorted_src, inv_group_rel, inv_group_dst, inv_mask,
         inv_src_group_ptr, inv_local_group_edge_start, inv_local_group_edge_count, inv_local_group_rel, inv_local_group_dst, inv_local_order, inv_local_group_src, inv_local_edge_group) = ctx.saved_tensors
        module = get_module()
        B, L, E = A.shape
        BL = B * L
        grad_A, grad_w = module.fastlog_backward_maxgroup_cuda(
            grad_ind.reshape(BL, E).contiguous().float(),
            grad_ori.reshape(BL, E).contiguous().float(),
            grad_inv.reshape(BL, E).contiguous().float(),
            ori_arg.contiguous(), inv_arg.contiguous(),
            A.reshape(BL, E).contiguous().float(),
            w.reshape(BL, -1).contiguous().float(),
            active_nodes.contiguous(),
            ori_sorted_src, ori_group_rel, ori_group_dst, ori_mask,
            ori_src_group_ptr, ori_local_group_edge_start, ori_local_group_edge_count, ori_local_group_rel, ori_local_group_dst, ori_local_order, ori_local_group_src, ori_local_edge_group,
            inv_sorted_src, inv_group_rel, inv_group_dst, inv_mask,
            inv_src_group_ptr, inv_local_group_edge_start, inv_local_group_edge_count, inv_local_group_rel, inv_local_group_dst, inv_local_order, inv_local_group_src, inv_local_edge_group,
            ctx.r_size, ctx.wot_i, ctx.use_topk, ctx.topk_edges
        )
        return (
            grad_A.reshape(B, L, E), grad_w.reshape(B, L, -1),
            None, None, None, None, None, None,
            None, None, None, None, None, None, None, None,
            None, None, None, None, None, None,
            None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None,
            None, None, None, None, None, None, None,
            None, None, None, None, None, None, None
        )


class FastLogFunctionSparse3DMaxGroupTopK(autograd.Function):
    @staticmethod
    def forward(ctx, sp_batch, sp_level, sp_entity, sp_value, w,
                ori_col_ind, ori_order, ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_col_ind, inv_order, inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                mask_values, weight, B, L, E, r_size, wot_i, topk_edges):
        module = get_module()
        mask_f = mask_values.float()
        edge_weight = weight if weight is not None else torch.empty(0, device=sp_value.device)

        (ori_b, ori_l, ori_e, ori_v,
         inv_b, inv_l, inv_e, inv_v,
         ind_b, ind_l, ind_e, ind_v,
         ori_meta_entry, ori_meta_rel, ori_meta_mask,
         inv_meta_entry, inv_meta_rel, inv_meta_mask,
         ori_offsets, inv_offsets,
         _ori_meta_edge, _inv_meta_edge) = module.fastlog_forward_sparse3d_topk_cuda(
            sp_batch, sp_level, sp_entity, sp_value.float(), w.float(),
            ori_col_ind.int(), ori_order.int(), ori_row_group_ptr.int(), ori_group_rel.short(),
            ori_group_edge_start.int(), ori_group_edge_count.int(),
            inv_col_ind.int(), inv_order.int(), inv_row_group_ptr.int(), inv_group_rel.short(),
            inv_group_edge_start.int(), inv_group_edge_count.int(),
            mask_f, edge_weight,
            B, L, E, r_size, wot_i, topk_edges, False, 1.0
        )

        out_ind, out_ori, out_inv, saved = _finalize_sparse3d_maxgroup_outputs(
            sp_batch, sp_level, sp_entity, sp_value, w,
            ori_b, ori_l, ori_e, ori_v, ori_meta_entry, ori_meta_rel, ori_meta_mask,
            inv_b, inv_l, inv_e, inv_v, inv_meta_entry, inv_meta_rel, inv_meta_mask,
            ind_b, ind_l, ind_e, ind_v,
            B, L, E, r_size
        )
        ctx.save_for_backward(*saved)
        ctx.B = B
        ctx.L = L
        ctx.E = E
        ctx.r_size = r_size
        ctx.wot_i = wot_i
        return out_ind, out_ori, out_inv

    @staticmethod
    def backward(ctx, grad_ind, grad_ori, grad_inv):
        grad_sp_value, grad_w = _backward_sparse3d_maxgroup_reduced(
            grad_ind, grad_ori, grad_inv, ctx.saved_tensors, ctx.L, ctx.E, ctx.r_size, ctx.wot_i
        )

        return (
            None, None, None, grad_sp_value, grad_w,
            None, None, None, None, None, None,
            None, None, None, None, None, None,
            None, None,
            None, None, None, None, None, None
        )


class FastLogFunctionSparse3DMaxGroupAllEdges(autograd.Function):
    @staticmethod
    def forward(ctx, sp_batch, sp_level, sp_entity, sp_value, w,
                ori_col_ind, ori_order, ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_col_ind, inv_order, inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                mask_values, weight, B, L, E, r_size, wot_i):
        module = get_module()
        mask_f = mask_values.float()
        w_tensor = weight.float().squeeze(-1) if weight is not None else torch.empty(0, device=sp_value.device)
        ori_mask = module.apply_mask_cuda(mask_f, ori_order.int(), w_tensor)
        inv_mask = module.apply_mask_cuda(mask_f, inv_order.int(), w_tensor)

        (ori_b, ori_l, ori_e, ori_v,
         inv_b, inv_l, inv_e, inv_v,
         ind_b, ind_l, ind_e, ind_v,
         ori_meta_entry, ori_meta_rel, ori_meta_mask,
         inv_meta_entry, inv_meta_rel, inv_meta_mask,
         ori_offsets, inv_offsets) = module.fastlog_forward_sparse3d_alledges_cuda(
            sp_batch, sp_level, sp_entity, sp_value.float(), w.float(),
            ori_col_ind, ori_mask, ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
            inv_col_ind, inv_mask, inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
            B, L, E, r_size, wot_i
        )

        out_ind, out_ori, out_inv, saved = _finalize_sparse3d_maxgroup_outputs(
            sp_batch, sp_level, sp_entity, sp_value, w,
            ori_b, ori_l, ori_e, ori_v, ori_meta_entry, ori_meta_rel, ori_meta_mask,
            inv_b, inv_l, inv_e, inv_v, inv_meta_entry, inv_meta_rel, inv_meta_mask,
            ind_b, ind_l, ind_e, ind_v,
            B, L, E, r_size
        )
        ctx.save_for_backward(*saved)
        ctx.B = B
        ctx.L = L
        ctx.E = E
        ctx.r_size = r_size
        ctx.wot_i = wot_i
        return out_ind, out_ori, out_inv

    @staticmethod
    def backward(ctx, grad_ind, grad_ori, grad_inv):
        grad_sp_value, grad_w = _backward_sparse3d_maxgroup_reduced(
            grad_ind, grad_ori, grad_inv, ctx.saved_tensors, ctx.L, ctx.E, ctx.r_size, ctx.wot_i
        )

        return (
            None, None, None, grad_sp_value, grad_w,
            None, None, None, None, None, None,
            None, None, None, None, None, None,
            None, None,
            None, None, None, None, None
        )

def vectorized_operation_csr_v2(A, w, csr_struct, mask_values, r_size,
                                 weight=None, wot_i=False,
                                 use_topk=False, topk_nodes=100000, topk_edges=0,
                                 agg_mode="sum"):
    pass                                              
                                                
                                                 
                                             
                                                 
                                                   

                                                                      
                                                                          
       
    (ori_row_ptr, ori_col_ind, ori_r_ind, order_ori,
     inv_row_ptr, inv_col_ind, inv_r_ind, order_inv,
     ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
     inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count) = csr_struct

    agg_id = 1 if agg_mode == "max" else 0
    return FastLogFunctionV2.apply(
        A, w,
        ori_row_ptr, ori_col_ind, ori_r_ind, order_ori,
        inv_row_ptr, inv_col_ind, inv_r_ind, order_inv,
        ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
        inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
        mask_values, weight,
        r_size, wot_i,
        use_topk, topk_nodes, topk_edges, agg_id
    )


def vectorized_operation_csr_v2_max_exact(A, w, csr_struct, mask_values, r_size,
                                          weight=None, wot_i=False,
                                          use_topk=False, topk_nodes=100000, topk_edges=0):
    (ori_row_ptr, ori_col_ind, ori_r_ind, order_ori,
     inv_row_ptr, inv_col_ind, inv_r_ind, order_inv,
     ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
     inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count) = csr_struct

    return FastLogFunctionV2MaxExact.apply(
        A, w,
        ori_row_ptr, ori_col_ind, ori_r_ind, order_ori,
        inv_row_ptr, inv_col_ind, inv_r_ind, order_inv,
        ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
        inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
        mask_values, weight,
        r_size, wot_i,
        use_topk, topk_nodes, topk_edges
    )


def vectorized_operation_maxgroup_csr_v2(A, w, csr_struct, mask_values, r_size,
                                         weight=None, wot_i=False,
                                         use_topk=False, topk_nodes=100000, topk_edges=0,
                                         smgroup_meta=None):
    ori_row_group_ptr = csr_struct[8]
    ori_group_rel = csr_struct[9]
    ori_group_edge_start = csr_struct[10]
    ori_group_edge_count = csr_struct[11]
    inv_row_group_ptr = csr_struct[12]
    inv_group_rel = csr_struct[13]
    inv_group_edge_start = csr_struct[14]
    inv_group_edge_count = csr_struct[15]
    if smgroup_meta is None:
        raise ValueError("smDRUM dense kernel requires smgroup metadata")
    return FastLogFunctionMaxGroupV2.apply(
        A, w,
        smgroup_meta[0], smgroup_meta[1], smgroup_meta[2], smgroup_meta[3], smgroup_meta[4], smgroup_meta[5],
        smgroup_meta[6], smgroup_meta[7], smgroup_meta[8], smgroup_meta[9], smgroup_meta[10], smgroup_meta[11], smgroup_meta[12], smgroup_meta[13],
        smgroup_meta[14], smgroup_meta[15], smgroup_meta[16], smgroup_meta[17], smgroup_meta[18], smgroup_meta[19],
        smgroup_meta[20], smgroup_meta[21], smgroup_meta[22], smgroup_meta[23], smgroup_meta[24], smgroup_meta[25], smgroup_meta[26], smgroup_meta[27],
        ori_row_group_ptr, ori_group_edge_start, ori_group_edge_count, ori_group_rel, smgroup_meta[28], smgroup_meta[29], smgroup_meta[30],
        inv_row_group_ptr, inv_group_edge_start, inv_group_edge_count, inv_group_rel, smgroup_meta[31], smgroup_meta[32], smgroup_meta[33],
        mask_values, weight, r_size, wot_i, use_topk, topk_nodes, topk_edges
    )


def _expand_sparse3d_direction_cpu(
    sp_batch, sp_level, sp_entity, sp_value, w,
    row_group_ptr, group_rel, group_edge_start, group_edge_count,
    col_ind, mask, rel_offset, topk_edges
):
    pass                                                                          
    nnz = sp_batch.shape[0]
    device = sp_batch.device
    dtype = sp_value.dtype

    empty = lambda: (torch.empty(0, dtype=torch.long, device=device),) * 3 + \
                     (torch.empty(0, dtype=dtype, device=device),) + \
                     (torch.empty(0, dtype=torch.long, device=device),
                      torch.empty(0, dtype=torch.long, device=device),
                      torch.empty(0, dtype=dtype, device=device))

    if nnz == 0:
        return empty()

                            
    g_starts = row_group_ptr[sp_entity].long()
    g_ends = row_group_ptr[sp_entity + 1].long()
    num_groups = g_ends - g_starts
    total_groups = num_groups.sum().item()

    if total_groups == 0:
        return empty()

                        
    entry_of_group = torch.repeat_interleave(torch.arange(nnz, device=device), num_groups)
    g_cumsum = torch.zeros(nnz + 1, dtype=torch.long, device=device)
    g_cumsum[1:] = num_groups.cumsum(0)
    local_g = torch.arange(total_groups, device=device) - g_cumsum[entry_of_group]
    group_idx = g_starts[entry_of_group] + local_g

    rels = group_rel[group_idx].long()
    edge_s = group_edge_start[group_idx].long()
    edge_c = group_edge_count[group_idx].long()

    b_g = sp_batch[entry_of_group]
    l_g = sp_level[entry_of_group]
    w_vals = w[b_g, l_g, rel_offset + rels]

                    
    takes = edge_c.clone()
    if topk_edges < (1 << 60):
        buckets = (w_vals.detach() * 255).clamp(0, 255).int()
        for i in range(nnz):
            gs = g_cumsum[i].item()
            ge = g_cumsum[i + 1].item()
            if gs == ge:
                continue
            total = edge_c[gs:ge].sum().item()
            if total <= topk_edges:
                continue
            hist = torch.zeros(256, dtype=torch.long, device=device)
            hist.scatter_add_(0, buckets[gs:ge].long(), edge_c[gs:ge])
            cumsum = 0
            threshold_bucket = -1
            keep_in_threshold = 0
            for bkt in range(255, -1, -1):
                c = hist[bkt].item()
                if cumsum + c >= topk_edges:
                    threshold_bucket = bkt
                    keep_in_threshold = topk_edges - cumsum
                    break
                cumsum += c
            if threshold_bucket < 0:
                continue
            threshold_seen = 0
            for j in range(gs, ge):
                bkt = buckets[j].item()
                if bkt < threshold_bucket:
                    takes[j] = 0
                elif bkt == threshold_bucket:
                    left = keep_in_threshold - threshold_seen
                    t = min(max(int(left), 0), int(takes[j].item()))
                    takes[j] = t
                    threshold_seen += t

                                                   
    valid = takes > 0
    if not valid.any():
        return empty()

    valid_idx = torch.where(valid)[0]
    valid_takes = takes[valid_idx]
    total_edges = valid_takes.sum().item()

    vg_of_edge = torch.repeat_interleave(torch.arange(valid_idx.shape[0], device=device), valid_takes)
    e_cumsum = torch.zeros(valid_idx.shape[0] + 1, dtype=torch.long, device=device)
    e_cumsum[1:] = valid_takes.cumsum(0)
    edge_local = torch.arange(total_edges, device=device) - e_cumsum[vg_of_edge]
    edge_pos = edge_s[valid_idx[vg_of_edge]] + edge_local

    dst = col_ind[edge_pos].long()
    m = mask[edge_pos].float()

                       
    nz = m != 0
    if not nz.all():
        vg_of_edge = vg_of_edge[nz]
        dst = dst[nz]
        m = m[nz]

    g_abs = valid_idx[vg_of_edge]
    eidx = entry_of_group[g_abs]
    out_b = sp_batch[eidx]
    out_l = sp_level[eidx]
    out_v = sp_value[eidx] * w_vals[g_abs] * m

    return out_b, out_l, dst, out_v, eidx, rels[g_abs], m


class FastLogFunctionSparse3DCPU(autograd.Function):
    pass                                                                               
    @staticmethod
    def forward(ctx, sp_batch, sp_level, sp_entity, sp_value, w,
                ori_col_ind, ori_mask,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_col_ind, inv_mask,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                B, L, E, r_size, wot_i, topk_edges):
        n = w.shape[2]
        device = sp_batch.device

        ori_b, ori_l, ori_e, ori_v, ori_entry, ori_rel, ori_m = \
            _expand_sparse3d_direction_cpu(
                sp_batch, sp_level, sp_entity, sp_value, w,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                ori_col_ind, ori_mask, 0, topk_edges)

        inv_b, inv_l, inv_e, inv_v, inv_entry, inv_rel, inv_m = \
            _expand_sparse3d_direction_cpu(
                sp_batch, sp_level, sp_entity, sp_value, w,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                inv_col_ind, inv_mask, r_size, topk_edges)

                  
        if not wot_i:
            ind_b, ind_l, ind_e = sp_batch, sp_level, sp_entity
            ind_v = sp_value * w[sp_batch, sp_level, n - 1]
        else:
            ind_b = ind_l = ind_e = torch.empty(0, dtype=torch.long, device=device)
            ind_v = torch.empty(0, dtype=sp_value.dtype, device=device)

                                                          
        ori_sp, _, _ = _coalesce_sparse3d_reduce(ori_b, ori_l, ori_e, ori_v, B, L, E)
        inv_sp, _, _ = _coalesce_sparse3d_reduce(inv_b, inv_l, inv_e, inv_v, B, L, E)
        ind_sp, _, _ = _coalesce_sparse3d_reduce(ind_b, ind_l, ind_e, ind_v, B, L, E)

        ctx.save_for_backward(
            sp_batch, sp_level, sp_entity, sp_value.float(), w.float(),
            ori_b, ori_l, ori_e, ori_entry.long(), ori_rel, ori_m,
            inv_b, inv_l, inv_e, inv_entry.long(), inv_rel, inv_m)
        ctx.B, ctx.L, ctx.E, ctx.r_size, ctx.wot_i = B, L, E, r_size, wot_i
        return ori_sp, inv_sp, ind_sp

    @staticmethod
    def backward(ctx, grad_ori, grad_inv, grad_ind):
        (sp_batch, sp_level, sp_entity, sp_value, w,
         ori_b, ori_l, ori_e, ori_entry, ori_rel, ori_m,
         inv_b, inv_l, inv_e, inv_entry, inv_rel, inv_m) = ctx.saved_tensors
        L, E, r_size = ctx.L, ctx.E, ctx.r_size
        n = w.shape[2]
        nnz = sp_batch.shape[0]

        grad_sp = torch.zeros(nnz, dtype=sp_value.dtype, device=sp_value.device)
        grad_w = torch.zeros_like(w)

                      
        if ori_b.numel() > 0:
            g = _gather_sparse3d_grad_at_entities(grad_ori, ori_b, ori_l, ori_e, L, E)
            wc = w[ori_b, ori_l, ori_rel]
            grad_sp.index_add_(0, ori_entry, g * wc * ori_m)
            widx = (ori_b * L + ori_l) * n + ori_rel
            grad_w.view(-1).index_add_(0, widx, g * sp_value[ori_entry] * ori_m)

                      
        if inv_b.numel() > 0:
            g = _gather_sparse3d_grad_at_entities(grad_inv, inv_b, inv_l, inv_e, L, E)
            rel_shifted = inv_rel + r_size
            wc = w[inv_b, inv_l, rel_shifted]
            grad_sp.index_add_(0, inv_entry, g * wc * inv_m)
            widx = (inv_b * L + inv_l) * n + rel_shifted
            grad_w.view(-1).index_add_(0, widx, g * sp_value[inv_entry] * inv_m)

                           
        if not ctx.wot_i:
            g = _gather_sparse3d_grad_at_entities(grad_ind, sp_batch, sp_level, sp_entity, L, E)
            grad_sp += g * w[sp_batch, sp_level, n - 1]
            widx = (sp_batch * L + sp_level) * n + (n - 1)
            grad_w.view(-1).index_add_(0, widx, g * sp_value)

        return (None, None, None, grad_sp, grad_w,
                None, None, None, None, None, None,
                None, None, None, None, None, None,
                None, None, None, None, None, None)


def vectorized_operation_sparse3d_csr(sp_state, w, csr_struct, mask_values,
                                       r_size, weight=None, wot_i=False,
                                       use_topk=False, topk_edges=0,
                                       agg_mode="sum"):
    pass                                                           
    sp_state = sp_state.coalesce()
    idx = sp_state.indices()
    val = sp_state.values()
    B, L, E = sp_state.shape

    ori_col_ind = csr_struct[1]
    order_ori = csr_struct[3]
    inv_col_ind = csr_struct[5]
    order_inv = csr_struct[7]
    ori_row_group_ptr = csr_struct[8]
    ori_group_rel = csr_struct[9]
    ori_group_edge_start = csr_struct[10]
    ori_group_edge_count = csr_struct[11]
    inv_row_group_ptr = csr_struct[12]
    inv_group_rel = csr_struct[13]
    inv_group_edge_start = csr_struct[14]
    inv_group_edge_count = csr_struct[15]

    effective_topk = topk_edges if (use_topk and topk_edges > 0) else (1 << 60)

    if sp_state.is_cuda:
                                                                                           
        ori_row_ptr = csr_struct[0]
        ori_r_ind = csr_struct[2]
        inv_row_ptr = csr_struct[4]
        inv_r_ind = csr_struct[6]
        return FastLogFunctionSparse3DTopK.apply(
            idx[0], idx[1], idx[2], val, w,
            ori_row_ptr, ori_col_ind, ori_r_ind, order_ori,
            inv_row_ptr, inv_col_ind, inv_r_ind, order_inv,
            ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
            inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
            mask_values, weight, B, L, E, r_size, wot_i, effective_topk,
            1 if agg_mode == "max" else 0)
    else:
                                        
        mask_f = mask_values.float()
        if weight is not None:
            mask_f = mask_f * score_function(weight).squeeze(-1)
        ori_mask = mask_f[order_ori]
        inv_mask = mask_f[order_inv]
        return FastLogFunctionSparse3DCPU.apply(
            idx[0], idx[1], idx[2], val, w,
            ori_col_ind, ori_mask,
            ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
            inv_col_ind, inv_mask,
            inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
            B, L, E, r_size, wot_i, effective_topk)


def vectorized_operation_maxgroup_sparse3d_csr_v2(sp_state, w, csr_struct, mask_values,
                                                  r_size, weight=None, wot_i=False,
                                                  use_topk=False, topk_edges=0):
    sp_state = sp_state.coalesce()
    idx = sp_state.indices()
    val = sp_state.values()
    B, L, E = sp_state.shape
    ori_col_ind = csr_struct[1]
    order_ori = csr_struct[3]
    inv_col_ind = csr_struct[5]
    order_inv = csr_struct[7]
    ori_row_group_ptr = csr_struct[8]
    ori_group_rel = csr_struct[9]
    ori_group_edge_start = csr_struct[10]
    ori_group_edge_count = csr_struct[11]
    inv_row_group_ptr = csr_struct[12]
    inv_group_rel = csr_struct[13]
    inv_group_edge_start = csr_struct[14]
    inv_group_edge_count = csr_struct[15]
    if use_topk and topk_edges > 0:
        return FastLogFunctionSparse3DMaxGroupTopK.apply(
            idx[0], idx[1], idx[2], val, w,
            ori_col_ind, order_ori, ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
            inv_col_ind, order_inv, inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
            mask_values, weight, B, L, E, r_size, wot_i, topk_edges
        )
    return FastLogFunctionSparse3DMaxGroupAllEdges.apply(
        idx[0], idx[1], idx[2], val, w,
        ori_col_ind, order_ori, ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
        inv_col_ind, order_inv, inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
        mask_values, weight, B, L, E, r_size, wot_i
    )
