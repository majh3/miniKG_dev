#include "fastlog.h"
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cstring>
#include <algorithm>

namespace fastlog {

Tensor ind2ptr(const Tensor &index, int64_t size) {
    Tensor idx = index.to(at::ScalarType::Long).contiguous();
    Tensor num = at::zeros({size}, idx.options());
    num.scatter_add_(0, idx, at::ones(idx.sizes(), idx.options()));
    Tensor ptr = num.cumsum(0) - num;
    Tensor total = at::tensor({index.size(0)}, ptr.options());
    return at::cat({ptr, total});
}

                                                                            
                                                                  
                                                                            
constexpr int NUM_BUCKETS = 256;

template <class scalar_t>
static inline int bucketize_weight_cpu(scalar_t score) {
    int bucket = static_cast<int>(score * static_cast<scalar_t>(NUM_BUCKETS - 1));
    return std::min(std::max(bucket, 0), NUM_BUCKETS - 1);
}

                                                             
                                                                        
template <class scalar_t>
static void forward_topk_row_cpu(
    int64_t src, int64_t bl, scalar_t a_val,
    const scalar_t *w,
    const int32_t *row_group_ptr,
    const int16_t *group_rel,
    const int32_t *group_edge_start,
    const int32_t *group_edge_count,
    const int32_t *col_ind,
    const scalar_t *mask,
    scalar_t *out,
    int64_t topk_edges,
    int64_t E, int64_t n, int64_t rel_offset
) {
    int64_t g_start = row_group_ptr[src];
    int64_t g_end   = row_group_ptr[src + 1];

                                                                
    int counts[NUM_BUCKETS] = {};
    int64_t total_edges = 0;
    for (int64_t g = g_start; g < g_end; ++g) {
        int64_t rel = group_rel[g];
        scalar_t score = w[bl * n + rel_offset + rel];
        int bucket = bucketize_weight_cpu(score);
        int32_t cnt = group_edge_count[g];
        counts[bucket] += cnt;
        total_edges += cnt;
    }

                                                               
    int threshold_bucket = -1;
    int64_t keep_in_threshold = 0;
    if (total_edges > topk_edges) {
        int64_t cumsum = 0;
        for (int b = NUM_BUCKETS - 1; b >= 0; --b) {
            if (cumsum + counts[b] >= topk_edges) {
                threshold_bucket = b;
                keep_in_threshold = topk_edges - cumsum;
                break;
            }
            cumsum += counts[b];
        }
    }

                                           
    int64_t threshold_seen = 0;
    for (int64_t g = g_start; g < g_end; ++g) {
        int64_t rel = group_rel[g];
        scalar_t w_val = w[bl * n + rel_offset + rel];
        int32_t take = group_edge_count[g];

        if (threshold_bucket >= 0) {
            int bucket = bucketize_weight_cpu(w_val);
            if (bucket < threshold_bucket) {
                take = 0;
            } else if (bucket == threshold_bucket) {
                int64_t left = keep_in_threshold - threshold_seen;
                take = left > 0
                    ? static_cast<int32_t>(std::min(left, static_cast<int64_t>(group_edge_count[g])))
                    : 0;
                threshold_seen += take;
            }
        }
        if (take <= 0) continue;

        int64_t edge_s = group_edge_start[g];
        for (int64_t p = edge_s; p < edge_s + take; ++p) {
            scalar_t m = mask[p];
            if (m == 0) continue;
            int64_t dst = col_ind[p];
            out[bl * E + dst] += a_val * w_val * m;
        }
    }
}

                                                                     
template <class scalar_t>
static void backward_topk_row_cpu(
    int64_t src, int64_t bl,
    const scalar_t *grad_out,
    const scalar_t *A, const scalar_t *w,
    const int32_t *row_group_ptr,
    const int16_t *group_rel,
    const int32_t *group_edge_start,
    const int32_t *group_edge_count,
    const int32_t *col_ind,
    const scalar_t *mask,
    scalar_t *gA, scalar_t *gw,
    int64_t topk_edges,
    int64_t E, int64_t n, int64_t rel_offset
) {
    int64_t g_start = row_group_ptr[src];
    int64_t g_end   = row_group_ptr[src + 1];

                                                              
    int counts[NUM_BUCKETS] = {};
    int64_t total_edges = 0;
    for (int64_t g = g_start; g < g_end; ++g) {
        int64_t rel = group_rel[g];
        scalar_t score = w[bl * n + rel_offset + rel];
        int bucket = bucketize_weight_cpu(score);
        counts[bucket] += group_edge_count[g];
        total_edges += group_edge_count[g];
    }

    int threshold_bucket = -1;
    int64_t keep_in_threshold = 0;
    if (total_edges > topk_edges) {
        int64_t cumsum = 0;
        for (int b = NUM_BUCKETS - 1; b >= 0; --b) {
            if (cumsum + counts[b] >= topk_edges) {
                threshold_bucket = b;
                keep_in_threshold = topk_edges - cumsum;
                break;
            }
            cumsum += counts[b];
        }
    }

                                                        
    scalar_t a_val = A[bl * E + src];
    int64_t threshold_seen = 0;
    for (int64_t g = g_start; g < g_end; ++g) {
        int64_t rel = group_rel[g];
        scalar_t w_val = w[bl * n + rel_offset + rel];
        int32_t take = group_edge_count[g];

        if (threshold_bucket >= 0) {
            int bucket = bucketize_weight_cpu(w_val);
            if (bucket < threshold_bucket) {
                take = 0;
            } else if (bucket == threshold_bucket) {
                int64_t left = keep_in_threshold - threshold_seen;
                take = left > 0
                    ? static_cast<int32_t>(std::min(left, static_cast<int64_t>(group_edge_count[g])))
                    : 0;
                threshold_seen += take;
            }
        }
        if (take <= 0) continue;

        int64_t edge_s = group_edge_start[g];
        for (int64_t p = edge_s; p < edge_s + take; ++p) {
            scalar_t m = mask[p];
            if (m == 0) continue;
            int64_t dst = col_ind[p];
            scalar_t go = grad_out[bl * E + dst];
            if (go == 0) continue;
            gA[bl * E + src] += go * w_val * m;
            gw[bl * n + rel_offset + rel] += go * a_val * m;
        }
    }
}

                                                                            
                                                    
                                                                            
template <class scalar_t>
void fastlog_forward_out_cpu(
    const scalar_t *A,                        
    const scalar_t *w,                        
    const int64_t  *active,                        
    int64_t         num_active,
    const int32_t  *ori_col_ind,
    const scalar_t *ori_mask,
    const int32_t  *ori_row_group_ptr,
    const int16_t  *ori_group_rel,
    const int32_t  *ori_group_edge_start,
    const int32_t  *ori_group_edge_count,
    const int32_t  *inv_col_ind,
    const scalar_t *inv_mask,
    const int32_t  *inv_row_group_ptr,
    const int16_t  *inv_group_rel,
    const int32_t  *inv_group_edge_start,
    const int32_t  *inv_group_edge_count,
    scalar_t *out_ind,
    scalar_t *out_ori,
    scalar_t *out_inv,
    int64_t BL, int64_t E, int64_t n,
    int64_t r_size, bool wot_i,
    int64_t topk_edges
) {
    int64_t BLE = BL * E;
    std::memset(out_ori, 0, BLE * sizeof(scalar_t));
    std::memset(out_inv, 0, BLE * sizeof(scalar_t));

                                                   
    if (!wot_i) {
        for (int64_t bl = 0; bl < BL; bl++) {
            scalar_t wid = w[bl * n + (n - 1)];
            const scalar_t *Abl = A + bl * E;
            scalar_t *obl = out_ind + bl * E;
            for (int64_t e = 0; e < E; e++)
                obl[e] = Abl[e] * wid;
        }
    } else {
        std::memset(out_ind, 0, BLE * sizeof(scalar_t));
    }

                                                          
    for (int64_t ai = 0; ai < num_active; ++ai) {
        int64_t src = active[ai];
        for (int64_t bl = 0; bl < BL; ++bl) {
            scalar_t a_val = A[bl * E + src];
            if (a_val == 0) continue;
            forward_topk_row_cpu(src, bl, a_val, w,
                ori_row_group_ptr, ori_group_rel,
                ori_group_edge_start, ori_group_edge_count,
                ori_col_ind, ori_mask, out_ori,
                topk_edges, E, n, 0);
            forward_topk_row_cpu(src, bl, a_val, w,
                inv_row_group_ptr, inv_group_rel,
                inv_group_edge_start, inv_group_edge_count,
                inv_col_ind, inv_mask, out_inv,
                topk_edges, E, n, r_size);
        }
    }
}

                                                                            
                                                     
                                                                            
template <class scalar_t>
void fastlog_backward_out_cpu(
    const scalar_t *grad_ind,
    const scalar_t *grad_ori,
    const scalar_t *grad_inv,
    const scalar_t *A,
    const scalar_t *w,
    const int64_t  *active,
    int64_t         num_active,
    const int32_t  *ori_col_ind,
    const scalar_t *ori_mask,
    const int32_t  *ori_row_group_ptr,
    const int16_t  *ori_group_rel,
    const int32_t  *ori_group_edge_start,
    const int32_t  *ori_group_edge_count,
    const int32_t  *inv_col_ind,
    const scalar_t *inv_mask,
    const int32_t  *inv_row_group_ptr,
    const int16_t  *inv_group_rel,
    const int32_t  *inv_group_edge_start,
    const int32_t  *inv_group_edge_count,
    scalar_t *gA, scalar_t *gw,
    int64_t BL, int64_t E, int64_t n,
    int64_t r_size, bool wot_i,
    int64_t topk_edges
) {
    int64_t BLE = BL * E;
    int64_t BLn = BL * n;
    std::memset(gA, 0, BLE * sizeof(scalar_t));
    std::memset(gw, 0, BLn * sizeof(scalar_t));

                        
    if (!wot_i) {
        for (int64_t bl = 0; bl < BL; bl++) {
            scalar_t wid = w[bl * n + (n - 1)];
            scalar_t gw_acc = 0;
            for (int64_t e = 0; e < E; e++) {
                scalar_t gi = grad_ind[bl * E + e];
                gA[bl * E + e] += gi * wid;
                gw_acc += gi * A[bl * E + e];
            }
            gw[bl * n + (n - 1)] += gw_acc;
        }
    }

                                            
    for (int64_t ai = 0; ai < num_active; ++ai) {
        int64_t src = active[ai];
        for (int64_t bl = 0; bl < BL; ++bl) {
            if (A[bl * E + src] == 0) continue;
            backward_topk_row_cpu(src, bl, grad_ori, A, w,
                ori_row_group_ptr, ori_group_rel,
                ori_group_edge_start, ori_group_edge_count,
                ori_col_ind, ori_mask, gA, gw,
                topk_edges, E, n, 0);
            backward_topk_row_cpu(src, bl, grad_inv, A, w,
                inv_row_group_ptr, inv_group_rel,
                inv_group_edge_start, inv_group_edge_count,
                inv_col_ind, inv_mask, gA, gw,
                topk_edges, E, n, r_size);
        }
    }
}

                                                                            
             
                                                                            

std::tuple<Tensor, Tensor, Tensor> fastlog_forward_cpu(
    const Tensor &A_, const Tensor &w_,
    const Tensor &active_nodes_,
    const Tensor &ori_col_ind_,   const Tensor &ori_mask_,
    const Tensor &ori_row_group_ptr_, const Tensor &ori_group_rel_,
    const Tensor &ori_group_edge_start_, const Tensor &ori_group_edge_count_,
    const Tensor &inv_col_ind_,   const Tensor &inv_mask_,
    const Tensor &inv_row_group_ptr_, const Tensor &inv_group_rel_,
    const Tensor &inv_group_edge_start_, const Tensor &inv_group_edge_count_,
    int64_t r_size, bool wot_i, int64_t topk_edges
) {
    TORCH_CHECK(A_.dim() == 2, "A must be 2D [BL, E]");
    TORCH_CHECK(w_.dim() == 2, "w must be 2D [BL, n]");

    const Tensor A    = A_.contiguous();
    const Tensor w    = w_.contiguous();
    const Tensor act  = active_nodes_.to(at::kLong).contiguous();
    const Tensor oci  = ori_col_ind_.to(at::kInt).contiguous();
    const Tensor om   = ori_mask_.to(A.dtype()).contiguous();
    const Tensor orgp = ori_row_group_ptr_.to(at::kInt).contiguous();
    const Tensor ogr  = ori_group_rel_.to(at::kShort).contiguous();
    const Tensor ogs  = ori_group_edge_start_.to(at::kInt).contiguous();
    const Tensor ogc  = ori_group_edge_count_.to(at::kInt).contiguous();
    const Tensor ici  = inv_col_ind_.to(at::kInt).contiguous();
    const Tensor im   = inv_mask_.to(A.dtype()).contiguous();
    const Tensor irgp = inv_row_group_ptr_.to(at::kInt).contiguous();
    const Tensor igr  = inv_group_rel_.to(at::kShort).contiguous();
    const Tensor igs  = inv_group_edge_start_.to(at::kInt).contiguous();
    const Tensor igc  = inv_group_edge_count_.to(at::kInt).contiguous();

    int64_t BL = A.size(0), E = A.size(1), n = w.size(1);
    Tensor out_ind = at::empty({BL, E}, A.options());
    Tensor out_ori = at::empty({BL, E}, A.options());
    Tensor out_inv = at::empty({BL, E}, A.options());

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "fastlog_forward_cpu", [&] {
        fastlog_forward_out_cpu<scalar_t>(
            A.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
            act.data_ptr<int64_t>(), act.size(0),
            oci.data_ptr<int32_t>(), om.data_ptr<scalar_t>(),
            orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(),
            ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
            ici.data_ptr<int32_t>(), im.data_ptr<scalar_t>(),
            irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(),
            igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
            out_ind.data_ptr<scalar_t>(),
            out_ori.data_ptr<scalar_t>(),
            out_inv.data_ptr<scalar_t>(),
            BL, E, n, r_size, wot_i, topk_edges
        );
    });

    return std::make_tuple(out_ind, out_ori, out_inv);
}

std::tuple<Tensor, Tensor> fastlog_backward_cpu(
    const Tensor &grad_ind_, const Tensor &grad_ori_, const Tensor &grad_inv_,
    const Tensor &A_, const Tensor &w_,
    const Tensor &active_nodes_,
    const Tensor &ori_col_ind_,   const Tensor &ori_mask_,
    const Tensor &ori_row_group_ptr_, const Tensor &ori_group_rel_,
    const Tensor &ori_group_edge_start_, const Tensor &ori_group_edge_count_,
    const Tensor &inv_col_ind_,   const Tensor &inv_mask_,
    const Tensor &inv_row_group_ptr_, const Tensor &inv_group_rel_,
    const Tensor &inv_group_edge_start_, const Tensor &inv_group_edge_count_,
    int64_t r_size, bool wot_i, int64_t topk_edges
) {
    const Tensor A   = A_.contiguous();
    const Tensor w   = w_.contiguous();
    const Tensor grad_ind = wot_i ? at::zeros_like(A) : grad_ind_.contiguous();
    const Tensor grad_ori = grad_ori_.contiguous();
    const Tensor grad_inv = grad_inv_.contiguous();
    const Tensor act  = active_nodes_.to(at::kLong).contiguous();
    const Tensor oci  = ori_col_ind_.to(at::kInt).contiguous();
    const Tensor om   = ori_mask_.to(A.dtype()).contiguous();
    const Tensor orgp = ori_row_group_ptr_.to(at::kInt).contiguous();
    const Tensor ogr  = ori_group_rel_.to(at::kShort).contiguous();
    const Tensor ogs  = ori_group_edge_start_.to(at::kInt).contiguous();
    const Tensor ogc  = ori_group_edge_count_.to(at::kInt).contiguous();
    const Tensor ici  = inv_col_ind_.to(at::kInt).contiguous();
    const Tensor im   = inv_mask_.to(A.dtype()).contiguous();
    const Tensor irgp = inv_row_group_ptr_.to(at::kInt).contiguous();
    const Tensor igr  = inv_group_rel_.to(at::kShort).contiguous();
    const Tensor igs  = inv_group_edge_start_.to(at::kInt).contiguous();
    const Tensor igc  = inv_group_edge_count_.to(at::kInt).contiguous();

    int64_t BL = A.size(0), E = A.size(1), n = w.size(1);
    Tensor gA = at::empty({BL, E}, A.options());
    Tensor gw = at::empty({BL, n}, w.options());

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "fastlog_backward_cpu", [&] {
        fastlog_backward_out_cpu<scalar_t>(
            grad_ind.data_ptr<scalar_t>(),
            grad_ori.data_ptr<scalar_t>(),
            grad_inv.data_ptr<scalar_t>(),
            A.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
            act.data_ptr<int64_t>(), act.size(0),
            oci.data_ptr<int32_t>(), om.data_ptr<scalar_t>(),
            orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(),
            ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
            ici.data_ptr<int32_t>(), im.data_ptr<scalar_t>(),
            irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(),
            igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
            gA.data_ptr<scalar_t>(), gw.data_ptr<scalar_t>(),
            BL, E, n, r_size, wot_i, topk_edges
        );
    });

    return std::make_tuple(gA, gw);
}

}                     

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ind2ptr",              &fastlog::ind2ptr);
    m.def("fastlog_forward_cpu",  &fastlog::fastlog_forward_cpu);
    m.def("fastlog_backward_cpu", &fastlog::fastlog_backward_cpu);
#ifdef WITH_CUDA
    m.def("fastlog_forward_cuda",  &fastlog::fastlog_forward_cuda);
    m.def("fastlog_backward_cuda", &fastlog::fastlog_backward_cuda);
    m.def("fastlog_forward_max_cuda", &fastlog::fastlog_forward_max_cuda);
    m.def("fastlog_backward_max_cuda", &fastlog::fastlog_backward_max_cuda);
    m.def("fastlog_forward_maxgroup_cuda", &fastlog::fastlog_forward_maxgroup_cuda);
    m.def("fastlog_backward_maxgroup_cuda", &fastlog::fastlog_backward_maxgroup_cuda);
    m.def("compute_active_nodes_cuda", &fastlog::compute_active_nodes_cuda);
    m.def("apply_mask_cuda",      &fastlog::apply_mask_cuda);
    m.def("fastlog_forward_sparse3d_topk_cuda",  &fastlog::fastlog_forward_sparse3d_topk_cuda);
    m.def("fastlog_forward_sparse3d_topk_masked_cuda",  &fastlog::fastlog_forward_sparse3d_topk_masked_cuda);
    m.def("fastlog_forward_sparse3d_alledges_cuda",  &fastlog::fastlog_forward_sparse3d_alledges_cuda);
    m.def("fastlog_backward_sparse3d_topk_cuda", &fastlog::fastlog_backward_sparse3d_topk_cuda);
    m.def("fastlog_backward_sparse3d_topk_aligned_cuda", &fastlog::fastlog_backward_sparse3d_topk_aligned_cuda);
#endif
}
