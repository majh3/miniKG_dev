#ifndef FASTLOG_H
#define FASTLOG_H

#include <torch/extension.h>
#include <tuple>

namespace fastlog {

using namespace torch;

// Convert sorted indices to row pointers (CSR format)
// Returns tensor of size [size+1]
Tensor ind2ptr(const Tensor &index, int64_t size);

// CSR group-based forward pass with active node filtering and top-k pruning.
// Uses relation groups (row_group_ptr, group_rel, group_edge_start, group_edge_count)
// instead of raw CSR row_ptr/r_ind — aligned with CUDA forward_topk_row.
// When topk_edges is very large (no pruning), all edges are kept.
//
// Returns: (output_ind, output_ori, output_inv)  each [BL, E]
std::tuple<Tensor, Tensor, Tensor> fastlog_forward_cpu(
    const Tensor &A,             // [BL, E] where BL = batch * L
    const Tensor &w,             // [BL, 2*r_size+1]
    const Tensor &active_nodes,  // [num_active] sorted unique active node indices
    const Tensor &ori_col_ind,   // [nnz]
    const Tensor &ori_mask,      // [nnz]
    const Tensor &ori_row_group_ptr,    // [E+1]
    const Tensor &ori_group_rel,        // [num_groups]
    const Tensor &ori_group_edge_start, // [num_groups]
    const Tensor &ori_group_edge_count, // [num_groups]
    const Tensor &inv_col_ind,   // [nnz]
    const Tensor &inv_mask,      // [nnz]
    const Tensor &inv_row_group_ptr,    // [E+1]
    const Tensor &inv_group_rel,        // [num_groups]
    const Tensor &inv_group_edge_start, // [num_groups]
    const Tensor &inv_group_edge_count, // [num_groups]
    int64_t r_size,
    bool wot_i,
    int64_t topk_edges
);

// CSR group-based backward pass with top-k pruning
// Returns: (grad_A, grad_w)  each [BL, ...]
std::tuple<Tensor, Tensor> fastlog_backward_cpu(
    const Tensor &grad_ind,      // [BL, E]
    const Tensor &grad_ori,      // [BL, E]
    const Tensor &grad_inv,      // [BL, E]
    const Tensor &A,             // [BL, E]
    const Tensor &w,             // [BL, n]
    const Tensor &active_nodes,
    const Tensor &ori_col_ind,
    const Tensor &ori_mask,
    const Tensor &ori_row_group_ptr,
    const Tensor &ori_group_rel,
    const Tensor &ori_group_edge_start,
    const Tensor &ori_group_edge_count,
    const Tensor &inv_col_ind,
    const Tensor &inv_mask,
    const Tensor &inv_row_group_ptr,
    const Tensor &inv_group_rel,
    const Tensor &inv_group_edge_start,
    const Tensor &inv_group_edge_count,
    int64_t r_size,
    bool wot_i,
    int64_t topk_edges
);

#ifdef WITH_CUDA
// Sparse3D top-k path: state is explicit COO over [B, L, E]
// Returns:
// ori_b, ori_l, ori_e, ori_v,
// inv_b, inv_l, inv_e, inv_v,
// ind_b, ind_l, ind_e, ind_v,
// ori_meta_entry, ori_meta_rel, ori_meta_mask,
// inv_meta_entry, inv_meta_rel, inv_meta_mask,
// ori_offsets, inv_offsets,
// ori_meta_edge, inv_meta_edge
std::tuple<Tensor,Tensor,Tensor,Tensor, Tensor,Tensor,Tensor,Tensor, Tensor,Tensor,Tensor,Tensor,
           Tensor,Tensor,Tensor, Tensor,Tensor,Tensor, Tensor,Tensor, Tensor,Tensor>
fastlog_forward_sparse3d_topk_cuda(
    const Tensor &sp_batch, const Tensor &sp_level, const Tensor &sp_entity, const Tensor &sp_value,
    const Tensor &w,
    const Tensor &ori_col_ind, const Tensor &ori_order,
    const Tensor &ori_row_group_ptr, const Tensor &ori_group_rel,
    const Tensor &ori_group_edge_start, const Tensor &ori_group_edge_count,
    const Tensor &inv_col_ind, const Tensor &inv_order,
    const Tensor &inv_row_group_ptr, const Tensor &inv_group_rel,
    const Tensor &inv_group_edge_start, const Tensor &inv_group_edge_count,
    const Tensor &mask_values, const Tensor &edge_weight,
    int64_t B, int64_t L, int64_t E, int64_t r_size, bool wot_i, int64_t topk_edges,
    bool edge_weight_is_score,
    double edge_weight_scale
);

std::tuple<Tensor,Tensor,Tensor,Tensor, Tensor,Tensor,Tensor,Tensor, Tensor,Tensor,Tensor,Tensor,
           Tensor,Tensor,Tensor, Tensor,Tensor,Tensor, Tensor,Tensor>
fastlog_forward_sparse3d_topk_masked_cuda(
    const Tensor &sp_batch, const Tensor &sp_level, const Tensor &sp_entity, const Tensor &sp_value,
    const Tensor &w,
    const Tensor &ori_col_ind, const Tensor &ori_mask,
    const Tensor &ori_row_group_ptr, const Tensor &ori_group_rel,
    const Tensor &ori_group_edge_start, const Tensor &ori_group_edge_count,
    const Tensor &inv_col_ind, const Tensor &inv_mask,
    const Tensor &inv_row_group_ptr, const Tensor &inv_group_rel,
    const Tensor &inv_group_edge_start, const Tensor &inv_group_edge_count,
    int64_t B, int64_t L, int64_t E, int64_t r_size, bool wot_i, int64_t topk_edges
);

std::tuple<Tensor,Tensor,Tensor,Tensor, Tensor,Tensor,Tensor,Tensor, Tensor,Tensor,Tensor,Tensor,
           Tensor,Tensor,Tensor, Tensor,Tensor,Tensor, Tensor,Tensor>
fastlog_forward_sparse3d_alledges_cuda(
    const Tensor &sp_batch, const Tensor &sp_level, const Tensor &sp_entity, const Tensor &sp_value,
    const Tensor &w,
    const Tensor &ori_col_ind, const Tensor &ori_mask,
    const Tensor &ori_row_group_ptr, const Tensor &ori_group_rel,
    const Tensor &ori_group_edge_start, const Tensor &ori_group_edge_count,
    const Tensor &inv_col_ind, const Tensor &inv_mask,
    const Tensor &inv_row_group_ptr, const Tensor &inv_group_rel,
    const Tensor &inv_group_edge_start, const Tensor &inv_group_edge_count,
    int64_t B, int64_t L, int64_t E, int64_t r_size, bool wot_i
);

std::tuple<Tensor, Tensor>
fastlog_backward_sparse3d_topk_cuda(
    const Tensor &grad_ori_values, const Tensor &grad_ori_batch, const Tensor &grad_ori_level, const Tensor &grad_ori_entity,
    const Tensor &grad_inv_values, const Tensor &grad_inv_batch, const Tensor &grad_inv_level, const Tensor &grad_inv_entity,
    const Tensor &grad_ind_values, const Tensor &grad_ind_batch, const Tensor &grad_ind_level, const Tensor &grad_ind_entity,
    const Tensor &sp_batch, const Tensor &sp_level, const Tensor &sp_entity, const Tensor &sp_value, const Tensor &w,
    const Tensor &ori_meta_batch, const Tensor &ori_meta_level, const Tensor &ori_meta_entity,
    const Tensor &ori_meta_entry, const Tensor &ori_meta_rel, const Tensor &ori_meta_mask, const Tensor &ori_offsets,
    const Tensor &inv_meta_batch, const Tensor &inv_meta_level, const Tensor &inv_meta_entity,
    const Tensor &inv_meta_entry, const Tensor &inv_meta_rel, const Tensor &inv_meta_mask, const Tensor &inv_offsets,
    int64_t B, int64_t L, int64_t E, int64_t r_size, bool wot_i
);

std::tuple<Tensor, Tensor>
fastlog_backward_sparse3d_topk_aligned_cuda(
    const Tensor &grad_ori_raw_values,
    const Tensor &ori_meta_entry, const Tensor &ori_meta_rel, const Tensor &ori_meta_mask,
    const Tensor &grad_inv_raw_values,
    const Tensor &inv_meta_entry, const Tensor &inv_meta_rel, const Tensor &inv_meta_mask,
    const Tensor &grad_ind_values, const Tensor &grad_ind_batch, const Tensor &grad_ind_level, const Tensor &grad_ind_entity,
    const Tensor &sp_batch, const Tensor &sp_level, const Tensor &sp_entity, const Tensor &sp_value, const Tensor &w,
    int64_t B, int64_t L, int64_t E, int64_t r_size, bool wot_i
);

std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>
fastlog_forward_maxgroup_cuda(
    const Tensor &A, const Tensor &w, const Tensor &active_nodes,
    const Tensor &ori_sorted_src, const Tensor &ori_group_edge_start,
    const Tensor &ori_group_edge_count, const Tensor &ori_group_rel,
    const Tensor &ori_group_dst, const Tensor &ori_mask,
    const Tensor &ori_src_group_ptr, const Tensor &ori_local_group_edge_start,
    const Tensor &ori_local_group_edge_count, const Tensor &ori_local_group_rel,
    const Tensor &ori_local_group_dst, const Tensor &ori_local_order,
    const Tensor &ori_local_group_src,
    const Tensor &ori_local_group_global,
    const Tensor &ori_row_group_ptr, const Tensor &ori_row_group_edge_start,
    const Tensor &ori_row_group_edge_count, const Tensor &ori_row_group_rel,
    const Tensor &ori_local_edge_group, const Tensor &ori_local_edge_row_group,
    const Tensor &ori_local_edge_row_offset,
    const Tensor &inv_sorted_src, const Tensor &inv_group_edge_start,
    const Tensor &inv_group_edge_count, const Tensor &inv_group_rel,
    const Tensor &inv_group_dst, const Tensor &inv_mask,
    const Tensor &inv_src_group_ptr, const Tensor &inv_local_group_edge_start,
    const Tensor &inv_local_group_edge_count, const Tensor &inv_local_group_rel,
    const Tensor &inv_local_group_dst, const Tensor &inv_local_order,
    const Tensor &inv_local_group_src,
    const Tensor &inv_local_group_global,
    const Tensor &inv_row_group_ptr, const Tensor &inv_row_group_edge_start,
    const Tensor &inv_row_group_edge_count, const Tensor &inv_row_group_rel,
    const Tensor &inv_local_edge_group, const Tensor &inv_local_edge_row_group,
    const Tensor &inv_local_edge_row_offset,
    int64_t r_size, bool wot_i, bool use_topk, int64_t topk_edges
);

std::tuple<Tensor,Tensor>
fastlog_backward_maxgroup_cuda(
    const Tensor &grad_ind, const Tensor &grad_ori, const Tensor &grad_inv,
    const Tensor &ori_arg, const Tensor &inv_arg,
    const Tensor &A, const Tensor &w, const Tensor &active_nodes,
    const Tensor &ori_sorted_src, const Tensor &ori_group_rel,
    const Tensor &ori_group_dst, const Tensor &ori_mask,
    const Tensor &ori_src_group_ptr, const Tensor &ori_local_group_edge_start,
    const Tensor &ori_local_group_edge_count, const Tensor &ori_local_group_rel,
    const Tensor &ori_local_group_dst, const Tensor &ori_local_order,
    const Tensor &ori_local_group_src, const Tensor &ori_local_edge_group,
    const Tensor &inv_sorted_src, const Tensor &inv_group_rel,
    const Tensor &inv_group_dst, const Tensor &inv_mask,
    const Tensor &inv_src_group_ptr, const Tensor &inv_local_group_edge_start,
    const Tensor &inv_local_group_edge_count, const Tensor &inv_local_group_rel,
    const Tensor &inv_local_group_dst, const Tensor &inv_local_order,
    const Tensor &inv_local_group_src, const Tensor &inv_local_edge_group,
    int64_t r_size, bool wot_i, bool use_topk, int64_t topk_edges
);

// Dense CUDA forward pass
std::tuple<Tensor, Tensor, Tensor> fastlog_forward_cuda(
    const Tensor &A, const Tensor &w, const Tensor &active_nodes,
    const Tensor &ori_row_ptr, const Tensor &ori_col_ind,
    const Tensor &ori_r_ind, const Tensor &ori_mask,
    const Tensor &inv_row_ptr, const Tensor &inv_col_ind,
    const Tensor &inv_r_ind, const Tensor &inv_mask,
    const Tensor &ori_row_group_ptr, const Tensor &ori_group_rel,
    const Tensor &ori_group_edge_start, const Tensor &ori_group_edge_count,
    const Tensor &inv_row_group_ptr, const Tensor &inv_group_rel,
    const Tensor &inv_group_edge_start, const Tensor &inv_group_edge_count,
    int64_t r_size, bool wot_i, bool use_topk, int64_t topk_edges, int64_t agg_mode
);

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> fastlog_forward_max_cuda(
    const Tensor &A, const Tensor &w, const Tensor &active_nodes,
    const Tensor &ori_row_ptr, const Tensor &ori_col_ind,
    const Tensor &ori_r_ind, const Tensor &ori_mask,
    const Tensor &inv_row_ptr, const Tensor &inv_col_ind,
    const Tensor &inv_r_ind, const Tensor &inv_mask,
    const Tensor &ori_row_group_ptr, const Tensor &ori_group_rel,
    const Tensor &ori_group_edge_start, const Tensor &ori_group_edge_count,
    const Tensor &inv_row_group_ptr, const Tensor &inv_group_rel,
    const Tensor &inv_group_edge_start, const Tensor &inv_group_edge_count,
    int64_t r_size, bool wot_i, bool use_topk, int64_t topk_edges
);

// CUDA backward pass
std::tuple<Tensor, Tensor> fastlog_backward_cuda(
    const Tensor &grad_ind, const Tensor &grad_ori, const Tensor &grad_inv,
    const Tensor &out_ori, const Tensor &out_inv,
    const Tensor &A, const Tensor &w, const Tensor &active_nodes,
    const Tensor &ori_row_ptr, const Tensor &ori_col_ind,
    const Tensor &ori_r_ind, const Tensor &ori_mask,
    const Tensor &inv_row_ptr, const Tensor &inv_col_ind,
    const Tensor &inv_r_ind, const Tensor &inv_mask,
    const Tensor &ori_row_group_ptr, const Tensor &ori_group_rel,
    const Tensor &ori_group_edge_start, const Tensor &ori_group_edge_count,
    const Tensor &inv_row_group_ptr, const Tensor &inv_group_rel,
    const Tensor &inv_group_edge_start, const Tensor &inv_group_edge_count,
    int64_t r_size, bool wot_i, bool use_topk, int64_t topk_edges, int64_t agg_mode
);

std::tuple<Tensor, Tensor> fastlog_backward_max_cuda(
    const Tensor &grad_ind, const Tensor &grad_ori, const Tensor &grad_inv,
    const Tensor &ori_arg, const Tensor &inv_arg,
    const Tensor &A, const Tensor &w, const Tensor &active_nodes,
    const Tensor &ori_row_ptr, const Tensor &ori_col_ind,
    const Tensor &ori_r_ind, const Tensor &ori_mask,
    const Tensor &inv_row_ptr, const Tensor &inv_col_ind,
    const Tensor &inv_r_ind, const Tensor &inv_mask,
    const Tensor &ori_row_group_ptr, const Tensor &ori_group_rel,
    const Tensor &ori_group_edge_start, const Tensor &ori_group_edge_count,
    const Tensor &inv_row_group_ptr, const Tensor &inv_group_rel,
    const Tensor &inv_group_edge_start, const Tensor &inv_group_edge_count,
    int64_t r_size, bool wot_i, bool use_topk, int64_t topk_edges
);


// Phase 2: GPU-side helpers
Tensor compute_active_nodes_cuda(const Tensor &A_flat);
Tensor apply_mask_cuda(const Tensor &mask_values, const Tensor &order, const Tensor &weight);
#endif

} // namespace fastlog

#endif // FASTLOG_H
