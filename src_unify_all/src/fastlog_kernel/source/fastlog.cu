#include "fastlog_cuda.cuh"
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <type_traits>

namespace fastlog {

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
constexpr int AGG_SUM = 0;
constexpr int AGG_MAX = 1;
constexpr int NUM_BUCKETS = 256;

template <typename scalar_t>
__device__ __forceinline__ int bucketize_weight(scalar_t score) {
    int bucket = static_cast<int>(score * static_cast<scalar_t>(NUM_BUCKETS - 1));
    return min(max(bucket, 0), NUM_BUCKETS - 1);
}

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    int* address_as_i = reinterpret_cast<int*>(addr);
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= value) break;
        old = atomicCAS(address_as_i, assumed, __float_as_int(value));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ __forceinline__ double atomicMaxDouble(double* addr, double value) {
    unsigned long long* address_as_ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(static_cast<long long>(assumed)) >= value) break;
        old = atomicCAS(address_as_ull, assumed,
                        static_cast<unsigned long long>(__double_as_longlong(value)));
    } while (assumed != old);
    return __longlong_as_double(static_cast<long long>(old));
}

template <typename scalar_t>
__device__ __forceinline__ void atomic_reduce(scalar_t* addr, scalar_t value, int64_t agg_mode) {
    if (agg_mode == AGG_MAX) {
        if constexpr (std::is_same<scalar_t, float>::value) {
            atomicMaxFloat(addr, value);
        } else if constexpr (std::is_same<scalar_t, double>::value) {
            atomicMaxDouble(addr, value);
        } else {
            atomicAdd(addr, value);
        }
    } else {
        atomicAdd(addr, value);
    }
}

__device__ __forceinline__ unsigned long long pack_max_pair(float value, int edge_idx) {
    return (static_cast<unsigned long long>(__float_as_uint(value)) << 32) |
           static_cast<unsigned int>(0xFFFFFFFFu - static_cast<unsigned int>(edge_idx));
}

__device__ __forceinline__ int unpack_max_edge(unsigned long long packed) {
    return static_cast<int>(0xFFFFFFFFu - static_cast<unsigned int>(packed & 0xFFFFFFFFu));
}

__device__ __forceinline__ void atomicMaxPair(unsigned long long* packed_ptr, float value, int edge_idx) {
    unsigned long long old = *packed_ptr, assumed;
    unsigned long long candidate = pack_max_pair(value, edge_idx);
    while (candidate > old) {
        assumed = old;
        old = atomicCAS(packed_ptr, assumed, candidate);
        if (old == assumed) break;
    }
}

__global__ void unpack_max_pairs_kernel(
    const unsigned long long* __restrict__ packed,
    float* __restrict__ out,
    int64_t* __restrict__ arg,
    int64_t total
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned long long v = packed[idx];
    out[idx] = __uint_as_float(static_cast<unsigned int>(v >> 32));
    int edge_idx = unpack_max_edge(v);
    arg[idx] = (v == 0ULL) ? -1 : static_cast<int64_t>(edge_idx);
}

constexpr int MAXGROUP_THREADS = 128;
constexpr int MAXGROUP_TOPK = 256;

template <typename scalar_t>
__global__ void fastlog_forward_maxgroup_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ w,
    const int32_t* __restrict__ sorted_src,
    const int32_t* __restrict__ group_edge_start,
    const int32_t* __restrict__ group_edge_count,
    const int16_t* __restrict__ group_rel,
    const int32_t* __restrict__ group_dst,
    const scalar_t* __restrict__ mask,
    scalar_t* __restrict__ out,
    int32_t* __restrict__ arg,
    int64_t num_groups,
    int64_t BL,
    int64_t E,
    int64_t n,
    int64_t rel_offset
) {
    int64_t g = blockIdx.x;
    int64_t bl = blockIdx.y;
    if (g >= num_groups || bl >= BL) return;

    __shared__ float sh_val[MAXGROUP_THREADS];
    __shared__ int sh_arg[MAXGROUP_THREADS];

    int tid = threadIdx.x;
    int start = group_edge_start[g];
    int count = group_edge_count[g];
    int rel = group_rel[g];
    float coeff = static_cast<float>(w[bl * n + rel_offset + rel]);
    float best = 0.0f;
    int best_p = -1;

    for (int e = tid; e < count; e += blockDim.x) {
        int p = start + e;
        int src = sorted_src[p];
        float v = static_cast<float>(A[bl * E + src]) * coeff * static_cast<float>(mask[p]);
        if (v > best || (v == best && (best_p < 0 || p < best_p))) {
            best = v;
            best_p = p;
        }
    }

    sh_val[tid] = best;
    sh_arg[tid] = best_p;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_v = sh_val[tid + stride];
            int other_p = sh_arg[tid + stride];
            if (other_v > sh_val[tid] || (other_v == sh_val[tid] && other_p >= 0 && (sh_arg[tid] < 0 || other_p < sh_arg[tid]))) {
                sh_val[tid] = other_v;
                sh_arg[tid] = other_p;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int32_t best_idx = sh_arg[0];
        arg[bl * num_groups + g] = best_idx;
        if (best_idx >= 0 && sh_val[0] != 0.0f) {
            atomicAdd(&out[bl * E + group_dst[g]], static_cast<scalar_t>(sh_val[0]));
        }
    }
}

template <typename scalar_t>
__global__ void fastlog_backward_maxgroup_kernel(
    const scalar_t* __restrict__ grad,
    const int32_t* __restrict__ arg,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ w,
    const int32_t* __restrict__ sorted_src,
    const int16_t* __restrict__ group_rel,
    const int32_t* __restrict__ group_dst,
    const scalar_t* __restrict__ mask,
    scalar_t* __restrict__ gA,
    scalar_t* __restrict__ gw,
    int64_t num_groups,
    int64_t BL,
    int64_t E,
    int64_t n,
    int64_t rel_offset
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = BL * num_groups;
    if (idx >= total) return;
    int64_t bl = idx / num_groups;
    int64_t g = idx % num_groups;
    int32_t p = arg[idx];
    if (p < 0) return;

    int dst = group_dst[g];
    scalar_t gg = grad[bl * E + dst];
    if (gg == 0) return;
    int rel = group_rel[g];
    int src = sorted_src[p];
    scalar_t m = mask[p];
    scalar_t wv = w[bl * n + rel_offset + rel];
    atomicAdd(&gA[bl * E + src], gg * wv * m);
    atomicAdd(&gw[bl * n + rel_offset + rel], gg * A[bl * E + src] * m);
}

template <typename scalar_t>
__global__ void fastlog_backward_maxgroup_topk_kernel(
    const scalar_t* __restrict__ grad,
    const int32_t* __restrict__ arg_local_pos,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ w,
    const int32_t* __restrict__ global_group_dst,
    const int16_t* __restrict__ local_group_rel,
    const int32_t* __restrict__ local_order,
    const int32_t* __restrict__ local_group_src,
    const int32_t* __restrict__ local_edge_group,
    const scalar_t* __restrict__ mask,
    scalar_t* __restrict__ gA,
    scalar_t* __restrict__ gw,
    int64_t num_groups,
    int64_t BL,
    int64_t E,
    int64_t n,
    int64_t rel_offset
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = BL * num_groups;
    if (idx >= total) return;
    int64_t bl = idx / num_groups;
    int64_t g = idx % num_groups;
    int32_t local_pos = arg_local_pos[idx];
    if (local_pos < 0) return;
    int32_t local_gid = local_edge_group[local_pos];
    int dst = global_group_dst[g];
    scalar_t gg = grad[bl * E + dst];
    if (gg == 0) return;
    int rel = local_group_rel[local_gid];
    int src = local_group_src[local_gid];
    scalar_t max_mask = mask[local_pos];
    if (max_mask == 0) return;
    scalar_t wv = w[bl * n + rel_offset + rel];
    atomicAdd(&gA[bl * E + src], gg * wv * max_mask);
    atomicAdd(&gw[bl * n + rel_offset + rel], gg * A[bl * E + src] * max_mask);
}

template <typename scalar_t>
__global__ void fastlog_forward_maxgroup_topk_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ w,
    const int32_t* __restrict__ row_group_ptr,
    const int32_t* __restrict__ row_group_edge_count,
    const int16_t* __restrict__ row_group_rel,
    const int32_t* __restrict__ src_group_ptr,
    const int32_t* __restrict__ local_group_edge_start,
    const int32_t* __restrict__ local_group_edge_count,
    const int16_t* __restrict__ local_group_rel,
    const int32_t* __restrict__ local_order,
    const int32_t* __restrict__ local_edge_row_group,
    const int32_t* __restrict__ local_edge_row_offset,
    const int32_t* __restrict__ local_group_global,
    const scalar_t* __restrict__ mask,
    unsigned long long* __restrict__ packed,
    const int64_t* __restrict__ active_nodes,
    int64_t num_active,
    int64_t num_groups,
    int64_t BL,
    int64_t E,
    int64_t n,
    int64_t rel_offset,
    int64_t topk_edges
) {
    int64_t node_idx = blockIdx.x;
    int64_t bl = blockIdx.y;
    if (node_idx >= num_active || bl >= BL) return;
    int64_t src = active_nodes[node_idx];
    float a = static_cast<float>(A[bl * E + src]);
    if (a == 0.0f) return;
    int64_t rg0 = row_group_ptr[src];
    int64_t rg1 = row_group_ptr[src + 1];
    __shared__ int shared_counts[NUM_BUCKETS];
    __shared__ int threshold_bucket_shared;
    __shared__ int64_t keep_in_threshold_shared;
    __shared__ int64_t active_edges_shared;
    __shared__ float sh_val[MAXGROUP_THREADS];
    __shared__ int sh_arg[MAXGROUP_THREADS];
    if (threadIdx.x == 0) {
        threshold_bucket_shared = -1;
        keep_in_threshold_shared = 0;
        active_edges_shared = 0;
    }
    __syncthreads();
    for (int b = threadIdx.x; b < NUM_BUCKETS; b += blockDim.x) shared_counts[b] = 0;
    __syncthreads();
    for (int64_t rg = rg0 + threadIdx.x; rg < rg1; rg += blockDim.x) {
        int bucket = bucketize_weight(w[bl * n + rel_offset + row_group_rel[rg]]);
        int32_t cnt = row_group_edge_count[rg];
        atomicAdd(&shared_counts[bucket], cnt);
        atomicAdd(reinterpret_cast<unsigned long long*>(&active_edges_shared), static_cast<unsigned long long>(cnt));
    }
    __syncthreads();
    if (threadIdx.x == 0 && active_edges_shared > topk_edges) {
        int64_t cumsum = 0;
        for (int b = NUM_BUCKETS - 1; b >= 0; --b) {
            int cnt = shared_counts[b];
            if (cumsum + cnt >= topk_edges) {
                threshold_bucket_shared = b;
                keep_in_threshold_shared = topk_edges - cumsum;
                break;
            }
            cumsum += cnt;
        }
    }
    __syncthreads();
    int64_t lg0 = src_group_ptr[src];
    int64_t lg1 = src_group_ptr[src + 1];
    for (int64_t g = lg0; g < lg1; ++g) {
        int rel = local_group_rel[g];
        float coeff = static_cast<float>(w[bl * n + rel_offset + rel]);
        float best_mask = 0.0f;
        int best_local_pos = -1;
        int start = local_group_edge_start[g];
        int count = local_group_edge_count[g];
        for (int e = threadIdx.x; e < count; e += blockDim.x) {
            int local_pos = start + e;
            int row_gid = local_edge_row_group[local_pos];
            int row_offset = local_edge_row_offset[local_pos];
            int bucket = bucketize_weight(w[bl * n + rel_offset + row_group_rel[row_gid]]);
            bool keep = true;
            if (threshold_bucket_shared >= 0) {
                if (bucket < threshold_bucket_shared) {
                    keep = false;
                } else if (bucket == threshold_bucket_shared) {
                    int64_t seen = 0;
                    for (int64_t rg = rg0; rg < row_gid; ++rg) {
                        if (bucketize_weight(w[bl * n + rel_offset + row_group_rel[rg]]) == threshold_bucket_shared) {
                            seen += row_group_edge_count[rg];
                        }
                    }
                    int64_t left = keep_in_threshold_shared - seen;
                    keep = left > 0 && row_offset < left;
                }
            }
            if (!keep) continue;
            float m = static_cast<float>(mask[local_pos]);
            if (m > best_mask || (m == best_mask && (best_local_pos < 0 || local_pos < best_local_pos))) {
                best_mask = m;
                best_local_pos = local_pos;
            }
        }
        sh_val[threadIdx.x] = best_mask;
        sh_arg[threadIdx.x] = best_local_pos;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                float ov = sh_val[threadIdx.x + stride];
                int op = sh_arg[threadIdx.x + stride];
                if (ov > sh_val[threadIdx.x] || (ov == sh_val[threadIdx.x] && op >= 0 && (sh_arg[threadIdx.x] < 0 || op < sh_arg[threadIdx.x]))) {
                    sh_val[threadIdx.x] = ov;
                    sh_arg[threadIdx.x] = op;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0 && sh_arg[0] >= 0 && sh_val[0] > 0.0f) {
            int global_gid = local_group_global[g];
            atomicMaxPair(&packed[bl * num_groups + global_gid], a * coeff * sh_val[0], sh_arg[0]);
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void accumulate_maxgroup_from_packed_kernel(
    const unsigned long long* __restrict__ packed,
    const int16_t* __restrict__ group_rel,
    const int32_t* __restrict__ group_dst,
    scalar_t* __restrict__ out,
    int32_t* __restrict__ arg,
    int64_t num_groups,
    int64_t BL,
    int64_t E
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = BL * num_groups;
    if (idx >= total) return;
    int64_t bl = idx / num_groups;
    int64_t g = idx % num_groups;
    unsigned long long v = packed[idx];
    int gid = unpack_max_edge(v);
    arg[idx] = (v == 0ULL) ? -1 : gid;
    if (v != 0ULL) {
        out[bl * E + group_dst[g]] += __uint_as_float(static_cast<unsigned int>(v >> 32));
    }
}

__global__ void mark_segment_start_kernel(
    const int64_t* __restrict__ sorted_keys,
    bool* __restrict__ flags,
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    flags[i] = (i == 0) || (sorted_keys[i] != sorted_keys[i - 1]);
}

template <typename scalar_t>
__global__ void reduce_segment_max_packed_kernel(
    const scalar_t* __restrict__ sorted_values,
    const int64_t* __restrict__ seg_id,
    unsigned long long* __restrict__ packed,
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicMaxPair(&packed[seg_id[i]], static_cast<float>(sorted_values[i]), static_cast<int>(i));
}

template <typename scalar_t>
__global__ void gather_segment_max_kernel(
    const unsigned long long* __restrict__ packed,
    const int64_t* __restrict__ seg_starts,
    const int64_t* __restrict__ sorted_batch,
    const int64_t* __restrict__ sorted_level,
    const int64_t* __restrict__ sorted_entity,
    const int16_t* __restrict__ sorted_rel,
    const int64_t* __restrict__ sorted_entry,
    const scalar_t* __restrict__ sorted_mask,
    const scalar_t* __restrict__ sorted_value,
    int64_t* __restrict__ red_batch,
    int64_t* __restrict__ red_level,
    int64_t* __restrict__ red_entity,
    int16_t* __restrict__ red_rel,
    int64_t* __restrict__ red_entry,
    scalar_t* __restrict__ red_mask,
    scalar_t* __restrict__ red_value,
    int64_t num_segments
) {
    int64_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_segments) return;
    int raw_idx = unpack_max_edge(packed[s]);
    if (raw_idx < 0) raw_idx = static_cast<int>(seg_starts[s]);
    red_batch[s] = sorted_batch[raw_idx];
    red_level[s] = sorted_level[raw_idx];
    red_entity[s] = sorted_entity[raw_idx];
    red_rel[s] = sorted_rel[raw_idx];
    red_entry[s] = sorted_entry[raw_idx];
    red_mask[s] = sorted_mask[raw_idx];
    red_value[s] = sorted_value[raw_idx];
}

template <typename scalar_t>
__global__ void sum_segments_to_output_kernel(
    const scalar_t* __restrict__ red_value,
    const int64_t* __restrict__ out_id,
    scalar_t* __restrict__ out_value,
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicAdd(&out_value[out_id[i]], red_value[i]);
}

__global__ void decode_output_keys_kernel(
    const int64_t* __restrict__ out_keys,
    int64_t* __restrict__ out_batch,
    int64_t* __restrict__ out_level,
    int64_t* __restrict__ out_entity,
    int64_t count,
    int64_t L,
    int64_t E
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int64_t key = out_keys[i];
    out_batch[i] = key / (L * E);
    int64_t rem = key % (L * E);
    out_level[i] = rem / E;
    out_entity[i] = rem % E;
}

template <typename scalar_t>
__global__ void fill_sparse3d_identity_kernel(
    const int64_t* __restrict__ sp_batch,
    const int64_t* __restrict__ sp_level,
    const int64_t* __restrict__ sp_entity,
    const scalar_t* __restrict__ sp_value,
    const scalar_t* __restrict__ w,
    int64_t* __restrict__ out_batch,
    int64_t* __restrict__ out_level,
    int64_t* __restrict__ out_entity,
    scalar_t* __restrict__ out_value,
    int64_t nnz,
    int64_t L,
    int64_t n
);

                                                                            
                                        
                                                    
                                                               
                                                           
                                                                            
template <typename scalar_t>
__global__ void fastlog_forward_fused_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ w,
    const int64_t* __restrict__ active_nodes,
              
    const int32_t* __restrict__ ori_row_ptr,
    const int32_t* __restrict__ ori_col_ind,
    const int16_t* __restrict__ ori_r_ind,
    const scalar_t* __restrict__ ori_mask,
              
    const int32_t* __restrict__ inv_row_ptr,
    const int32_t* __restrict__ inv_col_ind,
    const int16_t* __restrict__ inv_r_ind,
    const scalar_t* __restrict__ inv_mask,
              
    scalar_t* __restrict__ out_ori,
    scalar_t* __restrict__ out_inv,
    int64_t num_active,
    int64_t BL,
    int64_t E,
    int64_t n,
    int64_t r_size
) {
    extern __shared__ int32_t smem[];
    int32_t* col_buf = smem + threadIdx.y * WARP_SIZE;
    int32_t* rel_buf = smem + blockDim.y * WARP_SIZE + threadIdx.y * WARP_SIZE;
    scalar_t* mask_buf = reinterpret_cast<scalar_t*>(smem + 2 * blockDim.y * WARP_SIZE) + threadIdx.y * WARP_SIZE;

    int64_t node_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (node_idx >= num_active) return;

    int64_t src = active_nodes[node_idx];
    int64_t bl = blockIdx.y * WARP_SIZE + threadIdx.x;

                                                  
    scalar_t a_val = (bl < BL) ? A[bl * E + src] : 0;

                        
    {
        int64_t ptr_start = ori_row_ptr[src];
        int64_t ptr_end = ori_row_ptr[src + 1];

        for (int64_t block_ptr = ptr_start; block_ptr < ptr_end; block_ptr += WARP_SIZE) {
            int64_t ptr = block_ptr + threadIdx.x;
            if (ptr < ptr_end) {
                col_buf[threadIdx.x] = ori_col_ind[ptr];
                rel_buf[threadIdx.x] = ori_r_ind[ptr];
                mask_buf[threadIdx.x] = ori_mask[ptr];
            }
            __syncwarp();

            int64_t max_offset = ::min(static_cast<int64_t>(WARP_SIZE), ptr_end - block_ptr);
            if (bl < BL && a_val != 0) {
                for (int64_t i = 0; i < max_offset; i++) {
                    scalar_t m = mask_buf[i];
                    if (m == 0) continue;
                    int64_t dst = col_buf[i];
                    int64_t rel = rel_buf[i];
                    atomicAdd(&out_ori[bl * E + dst], a_val * w[bl * n + rel] * m);
                }
            }
            __syncwarp();
        }
    }

                                                           
    {
        int64_t ptr_start = inv_row_ptr[src];
        int64_t ptr_end = inv_row_ptr[src + 1];

        for (int64_t block_ptr = ptr_start; block_ptr < ptr_end; block_ptr += WARP_SIZE) {
            int64_t ptr = block_ptr + threadIdx.x;
            if (ptr < ptr_end) {
                col_buf[threadIdx.x] = inv_col_ind[ptr];
                rel_buf[threadIdx.x] = inv_r_ind[ptr];
                mask_buf[threadIdx.x] = inv_mask[ptr];
            }
            __syncwarp();

            int64_t max_offset = ::min(static_cast<int64_t>(WARP_SIZE), ptr_end - block_ptr);
            if (bl < BL && a_val != 0) {
                for (int64_t i = 0; i < max_offset; i++) {
                    scalar_t m = mask_buf[i];
                    if (m == 0) continue;
                    int64_t dst = col_buf[i];
                    int64_t rel = rel_buf[i];
                    atomicAdd(&out_inv[bl * E + dst], a_val * w[bl * n + r_size + rel] * m);
                }
            }
            __syncwarp();
        }
    }
}

                                      
template <typename scalar_t>
__global__ void fastlog_identity_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ w,
    scalar_t* __restrict__ out,
    int64_t BL,
    int64_t E,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = BL * E;
    if (idx >= total) return;

    int64_t bl = idx / E;
    out[idx] = A[idx] * w[bl * n + (n - 1)];
}

                                                                            
                                         
                                                      
                                                                            
template <typename scalar_t>
__global__ void fastlog_backward_fused_kernel(
    const scalar_t* __restrict__ grad_ori,
    const scalar_t* __restrict__ grad_inv,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ w,
    const int64_t* __restrict__ active_nodes,
              
    const int32_t* __restrict__ ori_row_ptr,
    const int32_t* __restrict__ ori_col_ind,
    const int16_t* __restrict__ ori_r_ind,
    const scalar_t* __restrict__ ori_mask,
              
    const int32_t* __restrict__ inv_row_ptr,
    const int32_t* __restrict__ inv_col_ind,
    const int16_t* __restrict__ inv_r_ind,
    const scalar_t* __restrict__ inv_mask,
              
    scalar_t* __restrict__ gA,
    scalar_t* __restrict__ gw,
    int64_t num_active,
    int64_t BL,
    int64_t E,
    int64_t n,
    int64_t r_size
) {
    extern __shared__ int32_t smem[];
    int32_t* col_buf = smem + threadIdx.y * WARP_SIZE;
    int32_t* rel_buf = smem + blockDim.y * WARP_SIZE + threadIdx.y * WARP_SIZE;
    scalar_t* mask_buf = reinterpret_cast<scalar_t*>(smem + 2 * blockDim.y * WARP_SIZE) + threadIdx.y * WARP_SIZE;

    int64_t node_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (node_idx >= num_active) return;

    int64_t src = active_nodes[node_idx];
    int64_t bl = blockIdx.y * WARP_SIZE + threadIdx.x;

    scalar_t a_val = (bl < BL) ? A[bl * E + src] : 0;

                           
    {
        int64_t ptr_start = ori_row_ptr[src];
        int64_t ptr_end = ori_row_ptr[src + 1];

        for (int64_t block_ptr = ptr_start; block_ptr < ptr_end; block_ptr += WARP_SIZE) {
            int64_t ptr = block_ptr + threadIdx.x;
            if (ptr < ptr_end) {
                col_buf[threadIdx.x] = ori_col_ind[ptr];
                rel_buf[threadIdx.x] = ori_r_ind[ptr];
                mask_buf[threadIdx.x] = ori_mask[ptr];
            }
            __syncwarp();

            int64_t max_offset = ::min(static_cast<int64_t>(WARP_SIZE), ptr_end - block_ptr);
            if (bl < BL) {
                for (int64_t i = 0; i < max_offset; i++) {
                    scalar_t m = mask_buf[i];
                    if (m == 0) continue;
                    int64_t dst = col_buf[i];
                    int64_t rel = rel_buf[i];
                    scalar_t go = grad_ori[bl * E + dst];
                    if (go == 0) continue;
                    scalar_t w_val = w[bl * n + rel];
                    atomicAdd(&gA[bl * E + src], go * w_val * m);
                    atomicAdd(&gw[bl * n + rel], go * a_val * m);
                }
            }
            __syncwarp();
        }
    }

                           
    {
        int64_t ptr_start = inv_row_ptr[src];
        int64_t ptr_end = inv_row_ptr[src + 1];

        for (int64_t block_ptr = ptr_start; block_ptr < ptr_end; block_ptr += WARP_SIZE) {
            int64_t ptr = block_ptr + threadIdx.x;
            if (ptr < ptr_end) {
                col_buf[threadIdx.x] = inv_col_ind[ptr];
                rel_buf[threadIdx.x] = inv_r_ind[ptr];
                mask_buf[threadIdx.x] = inv_mask[ptr];
            }
            __syncwarp();

            int64_t max_offset = ::min(static_cast<int64_t>(WARP_SIZE), ptr_end - block_ptr);
            if (bl < BL) {
                for (int64_t i = 0; i < max_offset; i++) {
                    scalar_t m = mask_buf[i];
                    if (m == 0) continue;
                    int64_t dst = col_buf[i];
                    int64_t rel = rel_buf[i];
                    scalar_t gi = grad_inv[bl * E + dst];
                    if (gi == 0) continue;
                    scalar_t w_val = w[bl * n + r_size + rel];
                    atomicAdd(&gA[bl * E + src], gi * w_val * m);
                    atomicAdd(&gw[bl * n + r_size + rel], gi * a_val * m);
                }
            }
            __syncwarp();
        }
    }
}

                                               
template <typename scalar_t>
__global__ void fastlog_identity_backward_kernel(
    const scalar_t* __restrict__ grad_ind,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ w,
    scalar_t* __restrict__ gA,
    scalar_t* __restrict__ gw,
    int64_t BL,
    int64_t E,
    int64_t n
) {
    int64_t bl = blockIdx.x * blockDim.x + threadIdx.x;
    if (bl >= BL) return;

    scalar_t wid = w[bl * n + (n - 1)];
    scalar_t gw_acc = 0;

    for (int64_t e = 0; e < E; e++) {
        scalar_t gi = grad_ind[bl * E + e];
        gA[bl * E + e] = gi * wid;
        gw_acc += gi * A[bl * E + e];
    }
    gw[bl * n + (n - 1)] = gw_acc;
}

constexpr int TOPK_THREADS_PER_BLOCK = 128;

__device__ __forceinline__ unsigned char* align_smem_ptr(unsigned char* ptr, size_t alignment) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);
    return reinterpret_cast<unsigned char*>(aligned);
}

template <typename scalar_t, bool IsInv>
__device__ void forward_topk_row(
    int64_t src,
    int64_t bl,
    scalar_t a_val,
    const scalar_t* __restrict__ w,
    const int32_t* __restrict__ row_group_ptr,
    const int16_t* __restrict__ group_rel,
    const int32_t* __restrict__ group_edge_start,
    const int32_t* __restrict__ group_edge_count,
    const int32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_ind,
    const int16_t* __restrict__ r_ind,
    const scalar_t* __restrict__ mask,
    scalar_t* __restrict__ out,
    int shared_counts[NUM_BUCKETS],
    int* threshold_bucket_shared,
    int64_t* keep_in_threshold_shared,
    int64_t* threshold_seen_shared,
    int64_t* active_edges_shared,
    int64_t topk_edges,
    int64_t E,
    int64_t n,
    int64_t rel_offset,
    int64_t agg_mode
) {
    int64_t group_start = row_group_ptr[src];
    int64_t group_end = row_group_ptr[src + 1];

    if (threadIdx.x == 0) {
        *threshold_bucket_shared = -1;
        *keep_in_threshold_shared = 0;
        *threshold_seen_shared = 0;
        *active_edges_shared = 0;
    }
    __syncthreads();

    for (int b = threadIdx.x; b < NUM_BUCKETS; b += blockDim.x) {
        shared_counts[b] = 0;
    }
    __syncthreads();

    for (int64_t g = group_start + threadIdx.x; g < group_end; g += blockDim.x) {
        int64_t rel = group_rel[g];
        scalar_t score = w[bl * n + rel_offset + rel];
        int bucket = bucketize_weight(score);
        int32_t cnt = group_edge_count[g];
        atomicAdd(&shared_counts[bucket], cnt);
        atomicAdd(reinterpret_cast<unsigned long long*>(active_edges_shared), static_cast<unsigned long long>(cnt));
    }
    __syncthreads();

    int threshold_bucket = NUM_BUCKETS - 1;
    int64_t keep_in_threshold = topk_edges;

    if (threadIdx.x == 0) {
        if (*active_edges_shared > topk_edges) {
            int64_t cumsum = 0;
            for (int b = NUM_BUCKETS - 1; b >= 0; --b) {
                int count = shared_counts[b];
                if (cumsum + count >= topk_edges) {
                    threshold_bucket = b;
                    keep_in_threshold = topk_edges - cumsum;
                    break;
                }
                cumsum += count;
            }
            *threshold_bucket_shared = threshold_bucket;
            *keep_in_threshold_shared = keep_in_threshold;
        } else {
            *threshold_bucket_shared = -1;
            *keep_in_threshold_shared = 0;
        }
    }
    __syncthreads();

    threshold_bucket = *threshold_bucket_shared;
    keep_in_threshold = *keep_in_threshold_shared;

    for (int64_t g = group_start; g < group_end; ++g) {
        int64_t rel = group_rel[g];
        scalar_t w_val = w[bl * n + rel_offset + rel];
        int32_t take = group_edge_count[g];
        if (threshold_bucket >= 0) {
            int bucket = bucketize_weight(w_val);
            if (bucket < threshold_bucket) {
                take = 0;
            } else if (bucket == threshold_bucket) {
                int64_t remain = atomicAdd(reinterpret_cast<unsigned long long*>(threshold_seen_shared), 0ULL);
                int64_t left = keep_in_threshold - remain;
                take = left > 0 ? static_cast<int32_t>(left < group_edge_count[g] ? left : group_edge_count[g]) : 0;
                if (threadIdx.x == 0 && take > 0) {
                    atomicAdd(reinterpret_cast<unsigned long long*>(threshold_seen_shared), static_cast<unsigned long long>(take));
                }
                __syncthreads();
            }
        }
        if (take <= 0) {
            __syncthreads();
            continue;
        }

        int64_t edge_start = group_edge_start[g];
        for (int64_t p = edge_start + threadIdx.x; p < edge_start + take; p += blockDim.x) {
            scalar_t m = mask[p];
            if (m == 0) continue;
            int64_t dst = col_ind[p];
            atomic_reduce(&out[bl * E + dst], a_val * w_val * m, agg_mode);
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void fastlog_forward_topk_fused_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ w,
    const int64_t* __restrict__ active_nodes,
    const int32_t* __restrict__ ori_row_ptr,
    const int32_t* __restrict__ ori_col_ind,
    const int16_t* __restrict__ ori_r_ind,
    const scalar_t* __restrict__ ori_mask,
    const int32_t* __restrict__ ori_row_group_ptr,
    const int16_t* __restrict__ ori_group_rel,
    const int32_t* __restrict__ ori_group_edge_start,
    const int32_t* __restrict__ ori_group_edge_count,
    const int32_t* __restrict__ inv_row_ptr,
    const int32_t* __restrict__ inv_col_ind,
    const int16_t* __restrict__ inv_r_ind,
    const scalar_t* __restrict__ inv_mask,
    const int32_t* __restrict__ inv_row_group_ptr,
    const int16_t* __restrict__ inv_group_rel,
    const int32_t* __restrict__ inv_group_edge_start,
    const int32_t* __restrict__ inv_group_edge_count,
    scalar_t* __restrict__ out_ori,
    scalar_t* __restrict__ out_inv,
    int64_t num_active,
    int64_t BL,
    int64_t E,
    int64_t n,
    int64_t r_size,
    int64_t topk_edges,
    int64_t agg_mode
) {
    int64_t node_idx = blockIdx.x;
    int64_t bl = blockIdx.y;
    if (node_idx >= num_active || bl >= BL) return;

    int64_t src = active_nodes[node_idx];
    scalar_t a_val = A[bl * E + src];
    if (a_val == 0) return;

    __shared__ int shared_counts[NUM_BUCKETS];
    __shared__ int threshold_bucket_shared;
    __shared__ int64_t keep_in_threshold_shared;
    __shared__ int64_t threshold_seen_shared;
    __shared__ int64_t active_edges_shared;
    forward_topk_row<scalar_t, false>(
        src, bl, a_val, w,
        ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
        ori_row_ptr, ori_col_ind, ori_r_ind, ori_mask,
        out_ori, shared_counts,
        &threshold_bucket_shared, &keep_in_threshold_shared,
        &threshold_seen_shared, &active_edges_shared,
        topk_edges, E, n, 0, agg_mode
    );
    __syncthreads();
    forward_topk_row<scalar_t, true>(
        src, bl, a_val, w,
        inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
        inv_row_ptr, inv_col_ind, inv_r_ind, inv_mask,
        out_inv, shared_counts,
        &threshold_bucket_shared, &keep_in_threshold_shared,
        &threshold_seen_shared, &active_edges_shared,
        topk_edges, E, n, r_size, agg_mode
    );
}

template <typename scalar_t>
__device__ void backward_topk_row(
    int64_t src,
    int64_t bl,
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ w,
    const int32_t* __restrict__ row_group_ptr,
    const int16_t* __restrict__ group_rel,
    const int32_t* __restrict__ group_edge_start,
    const int32_t* __restrict__ group_edge_count,
    const int32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_ind,
    const int16_t* __restrict__ r_ind,
    const scalar_t* __restrict__ mask,
    const scalar_t* __restrict__ out,
    scalar_t* __restrict__ gA,
    scalar_t* __restrict__ gw,
    int shared_counts[NUM_BUCKETS],
    int* threshold_bucket_shared,
    int64_t* keep_in_threshold_shared,
    int64_t* threshold_seen_shared,
    int64_t* active_edges_shared,
    int64_t topk_edges,
    int64_t E,
    int64_t n,
    int64_t rel_offset,
    int64_t agg_mode
) {
    int64_t group_start = row_group_ptr[src];
    int64_t group_end = row_group_ptr[src + 1];

    if (threadIdx.x == 0) {
        *threshold_bucket_shared = -1;
        *keep_in_threshold_shared = 0;
        *threshold_seen_shared = 0;
        *active_edges_shared = 0;
    }
    __syncthreads();

    for (int b = threadIdx.x; b < NUM_BUCKETS; b += blockDim.x) {
        shared_counts[b] = 0;
    }
    __syncthreads();

    for (int64_t g = group_start + threadIdx.x; g < group_end; g += blockDim.x) {
        int64_t rel = group_rel[g];
        scalar_t score = w[bl * n + rel_offset + rel];
        int bucket = bucketize_weight(score);
        int32_t cnt = group_edge_count[g];
        atomicAdd(&shared_counts[bucket], cnt);
        atomicAdd(reinterpret_cast<unsigned long long*>(active_edges_shared), static_cast<unsigned long long>(cnt));
    }
    __syncthreads();

    int threshold_bucket = NUM_BUCKETS - 1;
    int64_t keep_in_threshold = topk_edges;

    if (threadIdx.x == 0) {
        if (*active_edges_shared > topk_edges) {
            int64_t cumsum = 0;
            for (int b = NUM_BUCKETS - 1; b >= 0; --b) {
                int count = shared_counts[b];
                if (cumsum + count >= topk_edges) {
                    threshold_bucket = b;
                    keep_in_threshold = topk_edges - cumsum;
                    break;
                }
                cumsum += count;
            }
            *threshold_bucket_shared = threshold_bucket;
            *keep_in_threshold_shared = keep_in_threshold;
        } else {
            *threshold_bucket_shared = -1;
            *keep_in_threshold_shared = 0;
        }
    }
    __syncthreads();

    scalar_t a_val = A[bl * E + src];
    threshold_bucket = *threshold_bucket_shared;
    keep_in_threshold = *keep_in_threshold_shared;

    for (int64_t g = group_start; g < group_end; ++g) {
        int64_t rel = group_rel[g];
        scalar_t w_val = w[bl * n + rel_offset + rel];
        int32_t take = group_edge_count[g];
        if (threshold_bucket >= 0) {
            int bucket = bucketize_weight(w_val);
            if (bucket < threshold_bucket) {
                take = 0;
            } else if (bucket == threshold_bucket) {
                int64_t remain = atomicAdd(reinterpret_cast<unsigned long long*>(threshold_seen_shared), 0ULL);
                int64_t left = keep_in_threshold - remain;
                take = left > 0 ? static_cast<int32_t>(left < group_edge_count[g] ? left : group_edge_count[g]) : 0;
                if (threadIdx.x == 0 && take > 0) {
                    atomicAdd(reinterpret_cast<unsigned long long*>(threshold_seen_shared), static_cast<unsigned long long>(take));
                }
                __syncthreads();
            }
        }
        if (take <= 0) {
            __syncthreads();
            continue;
        }

        int64_t edge_start = group_edge_start[g];
        for (int64_t p = edge_start + threadIdx.x; p < edge_start + take; p += blockDim.x) {
            scalar_t m = mask[p];
            if (m == 0) continue;
            int64_t dst = col_ind[p];
            scalar_t go = grad_out[bl * E + dst];
            if (go == 0) continue;
            scalar_t contrib = a_val * w_val * m;
            if (agg_mode == AGG_MAX) {
                scalar_t out_val = out[bl * E + dst];
                scalar_t diff = contrib - out_val;
                if (diff < static_cast<scalar_t>(-1e-6) || diff > static_cast<scalar_t>(1e-6)) continue;
            }
            atomicAdd(&gA[bl * E + src], go * w_val * m);
            atomicAdd(&gw[bl * n + rel_offset + rel], go * a_val * m);
        }
        __syncthreads();
    }
}

__device__ void forward_max_topk_row(
    int64_t src,
    int64_t bl,
    float a_val,
    const float* __restrict__ w,
    const int32_t* __restrict__ row_group_ptr,
    const int16_t* __restrict__ group_rel,
    const int32_t* __restrict__ group_edge_start,
    const int32_t* __restrict__ group_edge_count,
    const int32_t* __restrict__ col_ind,
    const float* __restrict__ mask,
    unsigned long long* __restrict__ packed_out,
    int shared_counts[NUM_BUCKETS],
    int* threshold_bucket_shared,
    int64_t* keep_in_threshold_shared,
    int64_t* threshold_seen_shared,
    int64_t* active_edges_shared,
    int64_t topk_edges,
    int64_t E,
    int64_t n,
    int64_t rel_offset
) {
    int64_t group_start = row_group_ptr[src];
    int64_t group_end = row_group_ptr[src + 1];

    if (threadIdx.x == 0) {
        *threshold_bucket_shared = -1;
        *keep_in_threshold_shared = 0;
        *threshold_seen_shared = 0;
        *active_edges_shared = 0;
    }
    __syncthreads();

    for (int b = threadIdx.x; b < NUM_BUCKETS; b += blockDim.x) shared_counts[b] = 0;
    __syncthreads();

    for (int64_t g = group_start + threadIdx.x; g < group_end; g += blockDim.x) {
        int64_t rel = group_rel[g];
        int bucket = bucketize_weight(w[bl * n + rel_offset + rel]);
        int32_t cnt = group_edge_count[g];
        atomicAdd(&shared_counts[bucket], cnt);
        atomicAdd(reinterpret_cast<unsigned long long*>(active_edges_shared), static_cast<unsigned long long>(cnt));
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (*active_edges_shared > topk_edges) {
            int64_t cumsum = 0;
            for (int b = NUM_BUCKETS - 1; b >= 0; --b) {
                int count = shared_counts[b];
                if (cumsum + count >= topk_edges) {
                    *threshold_bucket_shared = b;
                    *keep_in_threshold_shared = topk_edges - cumsum;
                    break;
                }
                cumsum += count;
            }
        }
    }
    __syncthreads();

    int threshold_bucket = *threshold_bucket_shared;
    int64_t keep_in_threshold = *keep_in_threshold_shared;

    for (int64_t g = group_start; g < group_end; ++g) {
        int64_t rel = group_rel[g];
        float w_val = w[bl * n + rel_offset + rel];
        int32_t take = group_edge_count[g];
        if (threshold_bucket >= 0) {
            int bucket = bucketize_weight(w_val);
            if (bucket < threshold_bucket) {
                take = 0;
            } else if (bucket == threshold_bucket) {
                int64_t remain = atomicAdd(reinterpret_cast<unsigned long long*>(threshold_seen_shared), 0ULL);
                int64_t left = keep_in_threshold - remain;
                take = left > 0 ? static_cast<int32_t>(left < group_edge_count[g] ? left : group_edge_count[g]) : 0;
                if (threadIdx.x == 0 && take > 0) {
                    atomicAdd(reinterpret_cast<unsigned long long*>(threshold_seen_shared), static_cast<unsigned long long>(take));
                }
                __syncthreads();
            }
        }
        if (take <= 0) {
            __syncthreads();
            continue;
        }

        int64_t edge_start = group_edge_start[g];
        for (int64_t e = threadIdx.x; e < take; e += blockDim.x) {
            int64_t p = edge_start + e;
            float m = mask[p];
            if (m == 0) continue;
            int64_t dst = col_ind[p];
            float val = a_val * w_val * m;
            atomicMaxPair(&packed_out[bl * E + dst], val, static_cast<int>(p));
        }
        __syncthreads();
    }
}

__global__ void fastlog_forward_max_topk_fused_kernel(
    const float* __restrict__ A,
    const float* __restrict__ w,
    const int64_t* __restrict__ active_nodes,
    const int32_t* __restrict__ ori_col_ind,
    const float* __restrict__ ori_mask,
    const int32_t* __restrict__ ori_row_group_ptr,
    const int16_t* __restrict__ ori_group_rel,
    const int32_t* __restrict__ ori_group_edge_start,
    const int32_t* __restrict__ ori_group_edge_count,
    const int32_t* __restrict__ inv_col_ind,
    const float* __restrict__ inv_mask,
    const int32_t* __restrict__ inv_row_group_ptr,
    const int16_t* __restrict__ inv_group_rel,
    const int32_t* __restrict__ inv_group_edge_start,
    const int32_t* __restrict__ inv_group_edge_count,
    unsigned long long* __restrict__ out_ori_packed,
    unsigned long long* __restrict__ out_inv_packed,
    int64_t num_active,
    int64_t BL,
    int64_t E,
    int64_t n,
    int64_t r_size,
    int64_t topk_edges
) {
    int64_t node_idx = blockIdx.x;
    int64_t bl = blockIdx.y;
    if (node_idx >= num_active || bl >= BL) return;
    int64_t src = active_nodes[node_idx];
    float a_val = A[bl * E + src];
    if (a_val == 0) return;

    __shared__ int shared_counts[NUM_BUCKETS];
    __shared__ int threshold_bucket_shared;
    __shared__ int64_t keep_in_threshold_shared;
    __shared__ int64_t threshold_seen_shared;
    __shared__ int64_t active_edges_shared;
    forward_max_topk_row(
        src, bl, a_val, w,
        ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
        ori_col_ind, ori_mask, out_ori_packed,
        shared_counts, &threshold_bucket_shared, &keep_in_threshold_shared,
        &threshold_seen_shared, &active_edges_shared,
        topk_edges, E, n, 0
    );
    __syncthreads();
    forward_max_topk_row(
        src, bl, a_val, w,
        inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
        inv_col_ind, inv_mask, out_inv_packed,
        shared_counts, &threshold_bucket_shared, &keep_in_threshold_shared,
        &threshold_seen_shared, &active_edges_shared,
        topk_edges, E, n, r_size
    );
}

__global__ void fastlog_backward_max_topk_fused_kernel(
    const float* __restrict__ grad_ori,
    const float* __restrict__ grad_inv,
    const int64_t* __restrict__ ori_arg,
    const int64_t* __restrict__ inv_arg,
    const float* __restrict__ A,
    const float* __restrict__ w,
    const int64_t* __restrict__ active_nodes,
    const int32_t* __restrict__ ori_col_ind,
    const int16_t* __restrict__ ori_r_ind,
    const float* __restrict__ ori_mask,
    const int32_t* __restrict__ ori_row_group_ptr,
    const int16_t* __restrict__ ori_group_rel,
    const int32_t* __restrict__ ori_group_edge_start,
    const int32_t* __restrict__ ori_group_edge_count,
    const int32_t* __restrict__ inv_col_ind,
    const int16_t* __restrict__ inv_r_ind,
    const float* __restrict__ inv_mask,
    const int32_t* __restrict__ inv_row_group_ptr,
    const int16_t* __restrict__ inv_group_rel,
    const int32_t* __restrict__ inv_group_edge_start,
    const int32_t* __restrict__ inv_group_edge_count,
    float* __restrict__ gA,
    float* __restrict__ gw,
    int64_t num_active,
    int64_t BL,
    int64_t E,
    int64_t n,
    int64_t r_size,
    int64_t topk_edges
) {
    int64_t node_idx = blockIdx.x;
    int64_t bl = blockIdx.y;
    if (node_idx >= num_active || bl >= BL) return;
    int64_t src = active_nodes[node_idx];
    float a_val = A[bl * E + src];
    if (a_val == 0) return;

    __shared__ int shared_counts[NUM_BUCKETS];
    __shared__ int threshold_bucket_shared;
    __shared__ int64_t keep_in_threshold_shared;
    __shared__ int64_t threshold_seen_shared;
    __shared__ int64_t active_edges_shared;

    auto process = [&](const float* grad_out, const int64_t* arg_out,
                       const int32_t* col_ind, const int16_t* r_ind, const float* mask,
                       const int32_t* row_group_ptr, const int16_t* group_rel,
                       const int32_t* group_edge_start, const int32_t* group_edge_count,
                       int64_t rel_offset) {
        int64_t group_start = row_group_ptr[src];
        int64_t group_end = row_group_ptr[src + 1];
        if (threadIdx.x == 0) {
            threshold_bucket_shared = -1;
            keep_in_threshold_shared = 0;
            threshold_seen_shared = 0;
            active_edges_shared = 0;
        }
        __syncthreads();
        for (int b = threadIdx.x; b < NUM_BUCKETS; b += blockDim.x) shared_counts[b] = 0;
        __syncthreads();
        for (int64_t g = group_start + threadIdx.x; g < group_end; g += blockDim.x) {
            int64_t rel = group_rel[g];
            int bucket = bucketize_weight(w[bl * n + rel_offset + rel]);
            int32_t cnt = group_edge_count[g];
            atomicAdd(&shared_counts[bucket], cnt);
            atomicAdd(reinterpret_cast<unsigned long long*>(&active_edges_shared), static_cast<unsigned long long>(cnt));
        }
        __syncthreads();
        if (threadIdx.x == 0 && active_edges_shared > topk_edges) {
            int64_t cumsum = 0;
            for (int b = NUM_BUCKETS - 1; b >= 0; --b) {
                int count = shared_counts[b];
                if (cumsum + count >= topk_edges) {
                    threshold_bucket_shared = b;
                    keep_in_threshold_shared = topk_edges - cumsum;
                    break;
                }
                cumsum += count;
            }
        }
        __syncthreads();
        for (int64_t g = group_start; g < group_end; ++g) {
            int64_t rel = group_rel[g];
            float w_val = w[bl * n + rel_offset + rel];
            int32_t take = group_edge_count[g];
            if (threshold_bucket_shared >= 0) {
                int bucket = bucketize_weight(w_val);
                if (bucket < threshold_bucket_shared) {
                    take = 0;
                } else if (bucket == threshold_bucket_shared) {
                    int64_t remain = atomicAdd(reinterpret_cast<unsigned long long*>(&threshold_seen_shared), 0ULL);
                    int64_t left = keep_in_threshold_shared - remain;
                    take = left > 0 ? static_cast<int32_t>(left < group_edge_count[g] ? left : group_edge_count[g]) : 0;
                    if (threadIdx.x == 0 && take > 0) {
                        atomicAdd(reinterpret_cast<unsigned long long*>(&threshold_seen_shared), static_cast<unsigned long long>(take));
                    }
                    __syncthreads();
                }
            }
            if (take <= 0) {
                __syncthreads();
                continue;
            }
            int64_t edge_start = group_edge_start[g];
            for (int64_t e = threadIdx.x; e < take; e += blockDim.x) {
                int64_t p = edge_start + e;
                float m = mask[p];
                if (m == 0) continue;
                int64_t dst = col_ind[p];
                if (arg_out[bl * E + dst] != p) continue;
                float go = grad_out[bl * E + dst];
                if (go == 0) continue;
                atomicAdd(&gA[bl * E + src], go * w_val * m);
                atomicAdd(&gw[bl * n + rel_offset + rel], go * a_val * m);
            }
            __syncthreads();
        }
    };

    process(grad_ori, ori_arg, ori_col_ind, ori_r_ind, ori_mask,
            ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count, 0);
    __syncthreads();
    process(grad_inv, inv_arg, inv_col_ind, inv_r_ind, inv_mask,
            inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count, r_size);
}

template <typename scalar_t>
__global__ void fastlog_backward_topk_fused_kernel(
    const scalar_t* __restrict__ grad_ori,
    const scalar_t* __restrict__ grad_inv,
    const scalar_t* __restrict__ out_ori,
    const scalar_t* __restrict__ out_inv,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ w,
    const int64_t* __restrict__ active_nodes,
    const int32_t* __restrict__ ori_row_ptr,
    const int32_t* __restrict__ ori_col_ind,
    const int16_t* __restrict__ ori_r_ind,
    const scalar_t* __restrict__ ori_mask,
    const int32_t* __restrict__ ori_row_group_ptr,
    const int16_t* __restrict__ ori_group_rel,
    const int32_t* __restrict__ ori_group_edge_start,
    const int32_t* __restrict__ ori_group_edge_count,
    const int32_t* __restrict__ inv_row_ptr,
    const int32_t* __restrict__ inv_col_ind,
    const int16_t* __restrict__ inv_r_ind,
    const scalar_t* __restrict__ inv_mask,
    const int32_t* __restrict__ inv_row_group_ptr,
    const int16_t* __restrict__ inv_group_rel,
    const int32_t* __restrict__ inv_group_edge_start,
    const int32_t* __restrict__ inv_group_edge_count,
    scalar_t* __restrict__ gA,
    scalar_t* __restrict__ gw,
    int64_t num_active,
    int64_t BL,
    int64_t E,
    int64_t n,
    int64_t r_size,
    int64_t topk_edges,
    int64_t agg_mode
) {
    int64_t node_idx = blockIdx.x;
    int64_t bl = blockIdx.y;
    if (node_idx >= num_active || bl >= BL) return;

    int64_t src = active_nodes[node_idx];
    if (A[bl * E + src] == 0) return;

    __shared__ int shared_counts[NUM_BUCKETS];
    __shared__ int threshold_bucket_shared;
    __shared__ int64_t keep_in_threshold_shared;
    __shared__ int64_t threshold_seen_shared;
    __shared__ int64_t active_edges_shared;
    backward_topk_row(
        src, bl, grad_ori, A, w,
        ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
        ori_row_ptr, ori_col_ind, ori_r_ind, ori_mask, out_ori,
        gA, gw, shared_counts,
        &threshold_bucket_shared, &keep_in_threshold_shared,
        &threshold_seen_shared, &active_edges_shared,
        topk_edges, E, n, 0, agg_mode
    );
    __syncthreads();
    backward_topk_row(
        src, bl, grad_inv, A, w,
        inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
        inv_row_ptr, inv_col_ind, inv_r_ind, inv_mask, out_inv,
        gA, gw, shared_counts,
        &threshold_bucket_shared, &keep_in_threshold_shared,
        &threshold_seen_shared, &active_edges_shared,
        topk_edges, E, n, r_size, agg_mode
    );
}

                                                                            
                                                                   
                                                                            
std::tuple<Tensor, Tensor, Tensor> fastlog_forward_cuda(
    const Tensor &A_, const Tensor &w_,
    const Tensor &active_nodes_,
    const Tensor &ori_row_ptr_, const Tensor &ori_col_ind_,
    const Tensor &ori_r_ind_, const Tensor &ori_mask_,
    const Tensor &inv_row_ptr_, const Tensor &inv_col_ind_,
    const Tensor &inv_r_ind_, const Tensor &inv_mask_,
    const Tensor &ori_row_group_ptr_, const Tensor &ori_group_rel_,
    const Tensor &ori_group_edge_start_, const Tensor &ori_group_edge_count_,
    const Tensor &inv_row_group_ptr_, const Tensor &inv_group_rel_,
    const Tensor &inv_group_edge_start_, const Tensor &inv_group_edge_count_,
    int64_t r_size, bool wot_i, bool use_topk, int64_t topk_edges, int64_t agg_mode
) {
    TORCH_CHECK(A_.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(A_.dim() == 2, "A must be 2D [BL, E]");

    const Tensor A = A_.contiguous();
    const Tensor w = w_.contiguous();
    const Tensor act = active_nodes_.contiguous();
    const Tensor orp = ori_row_ptr_.contiguous();
    const Tensor oci = ori_col_ind_.contiguous();
    const Tensor ori = ori_r_ind_.contiguous();
    const Tensor om = ori_mask_.to(A.dtype()).contiguous();
    const Tensor irp = inv_row_ptr_.contiguous();
    const Tensor ici = inv_col_ind_.contiguous();
    const Tensor iri = inv_r_ind_.contiguous();
    const Tensor im = inv_mask_.to(A.dtype()).contiguous();
    const Tensor orgp = ori_row_group_ptr_.contiguous();
    const Tensor ogr = ori_group_rel_.contiguous();
    const Tensor ogs = ori_group_edge_start_.contiguous();
    const Tensor ogc = ori_group_edge_count_.contiguous();
    const Tensor irgp = inv_row_group_ptr_.contiguous();
    const Tensor igr = inv_group_rel_.contiguous();
    const Tensor igs = inv_group_edge_start_.contiguous();
    const Tensor igc = inv_group_edge_count_.contiguous();

    int64_t BL = A.size(0), E = A.size(1), n = w.size(1);
    int64_t num_active = act.size(0);

    Tensor out_ind = at::zeros({BL, E}, A.options());
    Tensor out_ori = at::zeros({BL, E}, A.options());
    Tensor out_inv = at::zeros({BL, E}, A.options());

                                     
    if (!wot_i) {
        int threads = 256;
        int blocks = (BL * E + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "fastlog_identity_cuda", [&] {
            fastlog_identity_kernel<scalar_t><<<blocks, threads>>>(
                A.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
                out_ind.data_ptr<scalar_t>(), BL, E, n
            );
        });
    }

                                                                              
                                                                             
                                  
    if (num_active > 0) {
        dim3 block(TOPK_THREADS_PER_BLOCK);
        dim3 grid(num_active, BL);
        int64_t effective_topk = (use_topk && topk_edges > 0) ? topk_edges : (static_cast<int64_t>(1) << 60);
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "fastlog_forward_group_fused_cuda", [&] {
            fastlog_forward_topk_fused_kernel<scalar_t><<<grid, block>>>(
                A.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
                act.data_ptr<int64_t>(),
                orp.data_ptr<int32_t>(), oci.data_ptr<int32_t>(),
                ori.data_ptr<int16_t>(), om.data_ptr<scalar_t>(),
                orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(),
                ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
                irp.data_ptr<int32_t>(), ici.data_ptr<int32_t>(),
                iri.data_ptr<int16_t>(), im.data_ptr<scalar_t>(),
                irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(),
                igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
                out_ori.data_ptr<scalar_t>(),
                out_inv.data_ptr<scalar_t>(),
                num_active, BL, E, n, r_size, effective_topk, agg_mode
            );
        });
    }

    return std::make_tuple(out_ind, out_ori, out_inv);
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> fastlog_forward_max_cuda(
    const Tensor &A_, const Tensor &w_,
    const Tensor &active_nodes_,
    const Tensor &ori_row_ptr_, const Tensor &ori_col_ind_,
    const Tensor &ori_r_ind_, const Tensor &ori_mask_,
    const Tensor &inv_row_ptr_, const Tensor &inv_col_ind_,
    const Tensor &inv_r_ind_, const Tensor &inv_mask_,
    const Tensor &ori_row_group_ptr_, const Tensor &ori_group_rel_,
    const Tensor &ori_group_edge_start_, const Tensor &ori_group_edge_count_,
    const Tensor &inv_row_group_ptr_, const Tensor &inv_group_rel_,
    const Tensor &inv_group_edge_start_, const Tensor &inv_group_edge_count_,
    int64_t r_size, bool wot_i, bool use_topk, int64_t topk_edges
) {
    const Tensor A = A_.contiguous().to(at::kFloat);
    const Tensor w = w_.contiguous().to(at::kFloat);
    const Tensor act = active_nodes_.contiguous();
    const Tensor oci = ori_col_ind_.contiguous();
    const Tensor ori = ori_r_ind_.contiguous();
    const Tensor om = ori_mask_.contiguous().to(at::kFloat);
    const Tensor ici = inv_col_ind_.contiguous();
    const Tensor iri = inv_r_ind_.contiguous();
    const Tensor im = inv_mask_.contiguous().to(at::kFloat);
    const Tensor orgp = ori_row_group_ptr_.contiguous();
    const Tensor ogr = ori_group_rel_.contiguous();
    const Tensor ogs = ori_group_edge_start_.contiguous();
    const Tensor ogc = ori_group_edge_count_.contiguous();
    const Tensor irgp = inv_row_group_ptr_.contiguous();
    const Tensor igr = inv_group_rel_.contiguous();
    const Tensor igs = inv_group_edge_start_.contiguous();
    const Tensor igc = inv_group_edge_count_.contiguous();

    int64_t BL = A.size(0), E = A.size(1), n = w.size(1);
    int64_t num_active = act.size(0);
    Tensor out_ind = at::zeros({BL, E}, A.options());
    Tensor out_ori = at::zeros({BL, E}, A.options());
    Tensor out_inv = at::zeros({BL, E}, A.options());
    Tensor ori_arg = at::full({BL, E}, -1, act.options());
    Tensor inv_arg = at::full({BL, E}, -1, act.options());
    Tensor out_ori_packed = at::zeros({BL, E}, act.options().dtype(at::kLong));
    Tensor out_inv_packed = at::zeros({BL, E}, act.options().dtype(at::kLong));

    if (!wot_i) {
        int threads = 256;
        int blocks = (BL * E + threads - 1) / threads;
        fastlog_identity_kernel<float><<<blocks, threads>>>(
            A.data_ptr<float>(), w.data_ptr<float>(),
            out_ind.data_ptr<float>(), BL, E, n
        );
    }
    if (num_active > 0) {
        dim3 block(TOPK_THREADS_PER_BLOCK);
        dim3 grid(num_active, BL);
        int64_t effective_topk = (use_topk && topk_edges > 0) ? topk_edges : (static_cast<int64_t>(1) << 60);
        fastlog_forward_max_topk_fused_kernel<<<grid, block>>>(
            A.data_ptr<float>(), w.data_ptr<float>(), act.data_ptr<int64_t>(),
            oci.data_ptr<int32_t>(), om.data_ptr<float>(),
            orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(), ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
            ici.data_ptr<int32_t>(), im.data_ptr<float>(),
            irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(), igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
            reinterpret_cast<unsigned long long*>(out_ori_packed.data_ptr<int64_t>()),
            reinterpret_cast<unsigned long long*>(out_inv_packed.data_ptr<int64_t>()),
            num_active, BL, E, n, r_size, effective_topk
        );
        int threads = 256;
        int blocks = (BL * E + threads - 1) / threads;
        unpack_max_pairs_kernel<<<blocks, threads>>>(
            reinterpret_cast<const unsigned long long*>(out_ori_packed.data_ptr<int64_t>()),
            out_ori.data_ptr<float>(), ori_arg.data_ptr<int64_t>(), BL * E
        );
        unpack_max_pairs_kernel<<<blocks, threads>>>(
            reinterpret_cast<const unsigned long long*>(out_inv_packed.data_ptr<int64_t>()),
            out_inv.data_ptr<float>(), inv_arg.data_ptr<int64_t>(), BL * E
        );
    }
    return std::make_tuple(out_ind, out_ori, out_inv, ori_arg, inv_arg);
}

                                                                            
                                                                            
                                                                            
std::tuple<Tensor, Tensor> fastlog_backward_cuda(
    const Tensor &grad_ind_, const Tensor &grad_ori_, const Tensor &grad_inv_,
    const Tensor &out_ori_, const Tensor &out_inv_,
    const Tensor &A_, const Tensor &w_,
    const Tensor &active_nodes_,
    const Tensor &ori_row_ptr_, const Tensor &ori_col_ind_,
    const Tensor &ori_r_ind_, const Tensor &ori_mask_,
    const Tensor &inv_row_ptr_, const Tensor &inv_col_ind_,
    const Tensor &inv_r_ind_, const Tensor &inv_mask_,
    const Tensor &ori_row_group_ptr_, const Tensor &ori_group_rel_,
    const Tensor &ori_group_edge_start_, const Tensor &ori_group_edge_count_,
    const Tensor &inv_row_group_ptr_, const Tensor &inv_group_rel_,
    const Tensor &inv_group_edge_start_, const Tensor &inv_group_edge_count_,
    int64_t r_size, bool wot_i, bool use_topk, int64_t topk_edges, int64_t agg_mode
) {
    const Tensor A = A_.contiguous();
    const Tensor w = w_.contiguous();
    const Tensor grad_ind = wot_i ? at::zeros_like(A) : grad_ind_.contiguous();
    const Tensor grad_ori = grad_ori_.contiguous();
    const Tensor grad_inv = grad_inv_.contiguous();
    const Tensor out_ori = out_ori_.contiguous();
    const Tensor out_inv = out_inv_.contiguous();
    const Tensor act = active_nodes_.contiguous();
    const Tensor orp = ori_row_ptr_.contiguous();
    const Tensor oci = ori_col_ind_.contiguous();
    const Tensor ori = ori_r_ind_.contiguous();
    const Tensor om = ori_mask_.to(A.dtype()).contiguous();
    const Tensor irp = inv_row_ptr_.contiguous();
    const Tensor ici = inv_col_ind_.contiguous();
    const Tensor iri = inv_r_ind_.contiguous();
    const Tensor im = inv_mask_.to(A.dtype()).contiguous();
    const Tensor orgp = ori_row_group_ptr_.contiguous();
    const Tensor ogr = ori_group_rel_.contiguous();
    const Tensor ogs = ori_group_edge_start_.contiguous();
    const Tensor ogc = ori_group_edge_count_.contiguous();
    const Tensor irgp = inv_row_group_ptr_.contiguous();
    const Tensor igr = inv_group_rel_.contiguous();
    const Tensor igs = inv_group_edge_start_.contiguous();
    const Tensor igc = inv_group_edge_count_.contiguous();

    int64_t BL = A.size(0), E = A.size(1), n = w.size(1);
    int64_t num_active = act.size(0);

    Tensor gA = at::zeros({BL, E}, A.options());
    Tensor gw = at::zeros({BL, n}, w.options());

                                              
    if (!wot_i) {
        int threads = 256;
        int blocks = (BL + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "fastlog_identity_backward_cuda", [&] {
            fastlog_identity_backward_kernel<scalar_t><<<blocks, threads>>>(
                grad_ind.data_ptr<scalar_t>(),
                A.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
                gA.data_ptr<scalar_t>(), gw.data_ptr<scalar_t>(),
                BL, E, n
            );
        });
    }

                                                                          
                                                                            
    if (num_active > 0) {
        dim3 block(TOPK_THREADS_PER_BLOCK);
        dim3 grid(num_active, BL);
        int64_t effective_topk = (use_topk && topk_edges > 0) ? topk_edges : (static_cast<int64_t>(1) << 60);
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "fastlog_backward_group_fused_cuda", [&] {
            fastlog_backward_topk_fused_kernel<scalar_t><<<grid, block>>>(
                grad_ori.data_ptr<scalar_t>(),
                grad_inv.data_ptr<scalar_t>(),
                out_ori.data_ptr<scalar_t>(),
                out_inv.data_ptr<scalar_t>(),
                A.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
                act.data_ptr<int64_t>(),
                orp.data_ptr<int32_t>(), oci.data_ptr<int32_t>(),
                ori.data_ptr<int16_t>(), om.data_ptr<scalar_t>(),
                orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(),
                ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
                irp.data_ptr<int32_t>(), ici.data_ptr<int32_t>(),
                iri.data_ptr<int16_t>(), im.data_ptr<scalar_t>(),
                irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(),
                igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
                gA.data_ptr<scalar_t>(), gw.data_ptr<scalar_t>(),
                num_active, BL, E, n, r_size, effective_topk, agg_mode
            );
        });
    }

    return std::make_tuple(gA, gw);
}

std::tuple<Tensor, Tensor> fastlog_backward_max_cuda(
    const Tensor &grad_ind_, const Tensor &grad_ori_, const Tensor &grad_inv_,
    const Tensor &ori_arg_, const Tensor &inv_arg_,
    const Tensor &A_, const Tensor &w_,
    const Tensor &active_nodes_,
    const Tensor &ori_row_ptr_, const Tensor &ori_col_ind_,
    const Tensor &ori_r_ind_, const Tensor &ori_mask_,
    const Tensor &inv_row_ptr_, const Tensor &inv_col_ind_,
    const Tensor &inv_r_ind_, const Tensor &inv_mask_,
    const Tensor &ori_row_group_ptr_, const Tensor &ori_group_rel_,
    const Tensor &ori_group_edge_start_, const Tensor &ori_group_edge_count_,
    const Tensor &inv_row_group_ptr_, const Tensor &inv_group_rel_,
    const Tensor &inv_group_edge_start_, const Tensor &inv_group_edge_count_,
    int64_t r_size, bool wot_i, bool use_topk, int64_t topk_edges
) {
    const Tensor A = A_.contiguous().to(at::kFloat);
    const Tensor w = w_.contiguous().to(at::kFloat);
    const Tensor grad_ind = wot_i ? at::zeros_like(A) : grad_ind_.contiguous().to(at::kFloat);
    const Tensor grad_ori = grad_ori_.contiguous().to(at::kFloat);
    const Tensor grad_inv = grad_inv_.contiguous().to(at::kFloat);
    const Tensor ori_arg = ori_arg_.contiguous();
    const Tensor inv_arg = inv_arg_.contiguous();
    const Tensor act = active_nodes_.contiguous();
    const Tensor oci = ori_col_ind_.contiguous();
    const Tensor ori = ori_r_ind_.contiguous();
    const Tensor om = ori_mask_.contiguous().to(at::kFloat);
    const Tensor ici = inv_col_ind_.contiguous();
    const Tensor iri = inv_r_ind_.contiguous();
    const Tensor im = inv_mask_.contiguous().to(at::kFloat);
    const Tensor orgp = ori_row_group_ptr_.contiguous();
    const Tensor ogr = ori_group_rel_.contiguous();
    const Tensor ogs = ori_group_edge_start_.contiguous();
    const Tensor ogc = ori_group_edge_count_.contiguous();
    const Tensor irgp = inv_row_group_ptr_.contiguous();
    const Tensor igr = inv_group_rel_.contiguous();
    const Tensor igs = inv_group_edge_start_.contiguous();
    const Tensor igc = inv_group_edge_count_.contiguous();

    int64_t BL = A.size(0), E = A.size(1), n = w.size(1);
    int64_t num_active = act.size(0);
    Tensor gA = at::zeros({BL, E}, A.options());
    Tensor gw = at::zeros({BL, n}, w.options());

    if (!wot_i) {
        int threads = 256;
        int blocks = (BL + threads - 1) / threads;
        fastlog_identity_backward_kernel<float><<<blocks, threads>>>(
            grad_ind.data_ptr<float>(), A.data_ptr<float>(), w.data_ptr<float>(),
            gA.data_ptr<float>(), gw.data_ptr<float>(), BL, E, n
        );
    }
    if (num_active > 0) {
        dim3 block(TOPK_THREADS_PER_BLOCK);
        dim3 grid(num_active, BL);
        int64_t effective_topk = (use_topk && topk_edges > 0) ? topk_edges : (static_cast<int64_t>(1) << 60);
        fastlog_backward_max_topk_fused_kernel<<<grid, block>>>(
            grad_ori.data_ptr<float>(), grad_inv.data_ptr<float>(),
            ori_arg.data_ptr<int64_t>(), inv_arg.data_ptr<int64_t>(),
            A.data_ptr<float>(), w.data_ptr<float>(), act.data_ptr<int64_t>(),
            oci.data_ptr<int32_t>(), ori.data_ptr<int16_t>(), om.data_ptr<float>(),
            orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(), ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
            ici.data_ptr<int32_t>(), iri.data_ptr<int16_t>(), im.data_ptr<float>(),
            irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(), igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
            gA.data_ptr<float>(), gw.data_ptr<float>(),
            num_active, BL, E, n, r_size, effective_topk
        );
    }
    return std::make_tuple(gA, gw);
}

std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> fastlog_forward_maxgroup_cuda(
    const Tensor &A_, const Tensor &w_, const Tensor &active_nodes_,
    const Tensor &ori_sorted_src_, const Tensor &ori_group_edge_start_,
    const Tensor &ori_group_edge_count_, const Tensor &ori_group_rel_,
    const Tensor &ori_group_dst_, const Tensor &ori_mask_,
    const Tensor &ori_src_group_ptr_, const Tensor &ori_local_group_edge_start_,
    const Tensor &ori_local_group_edge_count_, const Tensor &ori_local_group_rel_,
    const Tensor &ori_local_group_dst_, const Tensor &ori_local_order_,
    const Tensor &ori_local_group_src_,
    const Tensor &ori_local_group_global_,
    const Tensor &ori_row_group_ptr_, const Tensor &ori_row_group_edge_start_,
    const Tensor &ori_row_group_edge_count_, const Tensor &ori_row_group_rel_,
    const Tensor &ori_local_edge_group_, const Tensor &ori_local_edge_row_group_,
    const Tensor &ori_local_edge_row_offset_,
    const Tensor &inv_sorted_src_, const Tensor &inv_group_edge_start_,
    const Tensor &inv_group_edge_count_, const Tensor &inv_group_rel_,
    const Tensor &inv_group_dst_, const Tensor &inv_mask_,
    const Tensor &inv_src_group_ptr_, const Tensor &inv_local_group_edge_start_,
    const Tensor &inv_local_group_edge_count_, const Tensor &inv_local_group_rel_,
    const Tensor &inv_local_group_dst_, const Tensor &inv_local_order_,
    const Tensor &inv_local_group_src_,
    const Tensor &inv_local_group_global_,
    const Tensor &inv_row_group_ptr_, const Tensor &inv_row_group_edge_start_,
    const Tensor &inv_row_group_edge_count_, const Tensor &inv_row_group_rel_,
    const Tensor &inv_local_edge_group_, const Tensor &inv_local_edge_row_group_,
    const Tensor &inv_local_edge_row_offset_,
    int64_t r_size, bool wot_i, bool use_topk, int64_t topk_edges
) {
    const Tensor A = A_.contiguous().to(at::kFloat);
    const Tensor w = w_.contiguous().to(at::kFloat);
    const Tensor act = active_nodes_.contiguous();
    const Tensor oss = ori_sorted_src_.contiguous();
    const Tensor ogs = ori_group_edge_start_.contiguous();
    const Tensor ogc = ori_group_edge_count_.contiguous();
    const Tensor ogr = ori_group_rel_.contiguous();
    const Tensor ogd = ori_group_dst_.contiguous();
    const Tensor om = ori_mask_.contiguous().to(at::kFloat);
    const Tensor orsgp = ori_src_group_ptr_.contiguous();
    const Tensor olgs = ori_local_group_edge_start_.contiguous();
    const Tensor olgc = ori_local_group_edge_count_.contiguous();
    const Tensor olgr = ori_local_group_rel_.contiguous();
    const Tensor olgd = ori_local_group_dst_.contiguous();
    const Tensor olo = ori_local_order_.contiguous();
    const Tensor olsrc = ori_local_group_src_.contiguous();
    const Tensor olgg = ori_local_group_global_.contiguous();
    const Tensor orgp = ori_row_group_ptr_.contiguous();
    const Tensor orgec = ori_row_group_edge_count_.contiguous();
    const Tensor orgr = ori_row_group_rel_.contiguous();
    const Tensor oleg = ori_local_edge_group_.contiguous();
    const Tensor olerg = ori_local_edge_row_group_.contiguous();
    const Tensor olero = ori_local_edge_row_offset_.contiguous();
    const Tensor iss = inv_sorted_src_.contiguous();
    const Tensor igs = inv_group_edge_start_.contiguous();
    const Tensor igc = inv_group_edge_count_.contiguous();
    const Tensor igr = inv_group_rel_.contiguous();
    const Tensor igd = inv_group_dst_.contiguous();
    const Tensor im = inv_mask_.contiguous().to(at::kFloat);
    const Tensor irsgp = inv_src_group_ptr_.contiguous();
    const Tensor ilgs = inv_local_group_edge_start_.contiguous();
    const Tensor ilgc = inv_local_group_edge_count_.contiguous();
    const Tensor ilgr = inv_local_group_rel_.contiguous();
    const Tensor ilgd = inv_local_group_dst_.contiguous();
    const Tensor ilo = inv_local_order_.contiguous();
    const Tensor ilsrc = inv_local_group_src_.contiguous();
    const Tensor ilgg = inv_local_group_global_.contiguous();
    const Tensor irgp = inv_row_group_ptr_.contiguous();
    const Tensor irgec = inv_row_group_edge_count_.contiguous();
    const Tensor irgr = inv_row_group_rel_.contiguous();
    const Tensor ileg = inv_local_edge_group_.contiguous();
    const Tensor ilerg = inv_local_edge_row_group_.contiguous();
    const Tensor ilero = inv_local_edge_row_offset_.contiguous();

    int64_t BL = A.size(0), E = A.size(1), n = w.size(1);
    int64_t ori_groups = ogs.size(0);
    int64_t inv_groups = igs.size(0);
    Tensor out_ind = at::zeros({BL, E}, A.options());
    Tensor out_ori = at::zeros({BL, E}, A.options());
    Tensor out_inv = at::zeros({BL, E}, A.options());
    Tensor ori_arg = at::full({BL, ori_groups}, -1, oss.options().dtype(at::kInt));
    Tensor inv_arg = at::full({BL, inv_groups}, -1, iss.options().dtype(at::kInt));

    if (!wot_i) {
        int threads = 256;
        int blocks = (BL * E + threads - 1) / threads;
        fastlog_identity_kernel<float><<<blocks, threads>>>(
            A.data_ptr<float>(), w.data_ptr<float>(), out_ind.data_ptr<float>(), BL, E, n
        );
    }
    if (!use_topk) {
        if (ori_groups > 0) {
            dim3 grid(ori_groups, BL);
            fastlog_forward_maxgroup_kernel<float><<<grid, MAXGROUP_THREADS>>>(
                A.data_ptr<float>(), w.data_ptr<float>(),
                oss.data_ptr<int32_t>(), ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
                ogr.data_ptr<int16_t>(), ogd.data_ptr<int32_t>(), om.data_ptr<float>(),
                out_ori.data_ptr<float>(), ori_arg.data_ptr<int32_t>(),
                ori_groups, BL, E, n, 0
            );
        }
        if (inv_groups > 0) {
            dim3 grid(inv_groups, BL);
            fastlog_forward_maxgroup_kernel<float><<<grid, MAXGROUP_THREADS>>>(
                A.data_ptr<float>(), w.data_ptr<float>(),
                iss.data_ptr<int32_t>(), igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
                igr.data_ptr<int16_t>(), igd.data_ptr<int32_t>(), im.data_ptr<float>(),
                out_inv.data_ptr<float>(), inv_arg.data_ptr<int32_t>(),
                inv_groups, BL, E, n, r_size
            );
        }
    } else {
        Tensor ori_packed = at::zeros({BL, ori_groups}, A.options().dtype(at::kLong));
        Tensor inv_packed = at::zeros({BL, inv_groups}, A.options().dtype(at::kLong));
        int64_t num_active = act.size(0);
        if (orsgp.numel() > 0 && num_active > 0) {
            dim3 grid(num_active, BL);
            fastlog_forward_maxgroup_topk_kernel<float><<<grid, MAXGROUP_THREADS>>>(
                A.data_ptr<float>(), w.data_ptr<float>(),
                orgp.data_ptr<int32_t>(), orgec.data_ptr<int32_t>(), orgr.data_ptr<int16_t>(),
                orsgp.data_ptr<int32_t>(),
                olgs.data_ptr<int32_t>(), olgc.data_ptr<int32_t>(),
                olgr.data_ptr<int16_t>(), olo.data_ptr<int32_t>(), olerg.data_ptr<int32_t>(), olero.data_ptr<int32_t>(), olgg.data_ptr<int32_t>(), om.data_ptr<float>(),
                reinterpret_cast<unsigned long long*>(ori_packed.data_ptr<int64_t>()),
                act.data_ptr<int64_t>(), num_active, ori_groups, BL, E, n, 0, topk_edges
            );
        }
        if (irsgp.numel() > 0 && num_active > 0) {
            dim3 grid(num_active, BL);
            fastlog_forward_maxgroup_topk_kernel<float><<<grid, MAXGROUP_THREADS>>>(
                A.data_ptr<float>(), w.data_ptr<float>(),
                irgp.data_ptr<int32_t>(), irgec.data_ptr<int32_t>(), irgr.data_ptr<int16_t>(),
                irsgp.data_ptr<int32_t>(),
                ilgs.data_ptr<int32_t>(), ilgc.data_ptr<int32_t>(),
                ilgr.data_ptr<int16_t>(), ilo.data_ptr<int32_t>(), ilerg.data_ptr<int32_t>(), ilero.data_ptr<int32_t>(), ilgg.data_ptr<int32_t>(), im.data_ptr<float>(),
                reinterpret_cast<unsigned long long*>(inv_packed.data_ptr<int64_t>()),
                act.data_ptr<int64_t>(), num_active, inv_groups, BL, E, n, r_size, topk_edges
            );
        }
        int threads = 256;
        if (ori_groups > 0) {
            int blocks = static_cast<int>((BL * ori_groups + threads - 1) / threads);
            accumulate_maxgroup_from_packed_kernel<float><<<blocks, threads>>>(
                reinterpret_cast<const unsigned long long*>(ori_packed.data_ptr<int64_t>()),
                ogr.data_ptr<int16_t>(), ogd.data_ptr<int32_t>(),
                out_ori.data_ptr<float>(), ori_arg.data_ptr<int32_t>(),
                ori_groups, BL, E
            );
        }
        if (inv_groups > 0) {
            int blocks = static_cast<int>((BL * inv_groups + threads - 1) / threads);
            accumulate_maxgroup_from_packed_kernel<float><<<blocks, threads>>>(
                reinterpret_cast<const unsigned long long*>(inv_packed.data_ptr<int64_t>()),
                igr.data_ptr<int16_t>(), igd.data_ptr<int32_t>(),
                out_inv.data_ptr<float>(), inv_arg.data_ptr<int32_t>(),
                inv_groups, BL, E
            );
        }
    }
    return std::make_tuple(out_ind, out_ori, out_inv, ori_arg, inv_arg);
}

std::tuple<Tensor,Tensor> fastlog_backward_maxgroup_cuda(
    const Tensor &grad_ind_, const Tensor &grad_ori_, const Tensor &grad_inv_,
    const Tensor &ori_arg_, const Tensor &inv_arg_,
    const Tensor &A_, const Tensor &w_, const Tensor &active_nodes_,
    const Tensor &ori_sorted_src_, const Tensor &ori_group_rel_,
    const Tensor &ori_group_dst_, const Tensor &ori_mask_,
    const Tensor &ori_src_group_ptr_, const Tensor &ori_local_group_edge_start_,
    const Tensor &ori_local_group_edge_count_, const Tensor &ori_local_group_rel_,
    const Tensor &ori_local_group_dst_, const Tensor &ori_local_order_,
    const Tensor &ori_local_group_src_, const Tensor &ori_local_edge_group_,
    const Tensor &inv_sorted_src_, const Tensor &inv_group_rel_,
    const Tensor &inv_group_dst_, const Tensor &inv_mask_,
    const Tensor &inv_src_group_ptr_, const Tensor &inv_local_group_edge_start_,
    const Tensor &inv_local_group_edge_count_, const Tensor &inv_local_group_rel_,
    const Tensor &inv_local_group_dst_, const Tensor &inv_local_order_,
    const Tensor &inv_local_group_src_, const Tensor &inv_local_edge_group_,
    int64_t r_size, bool wot_i, bool use_topk, int64_t topk_edges
) {
    const Tensor A = A_.contiguous().to(at::kFloat);
    const Tensor w = w_.contiguous().to(at::kFloat);
    const Tensor act = active_nodes_.contiguous();
    const Tensor grad_ind = wot_i ? at::zeros_like(A) : grad_ind_.contiguous().to(at::kFloat);
    const Tensor grad_ori = grad_ori_.contiguous().to(at::kFloat);
    const Tensor grad_inv = grad_inv_.contiguous().to(at::kFloat);
    const Tensor ori_arg = ori_arg_.contiguous();
    const Tensor inv_arg = inv_arg_.contiguous();
    const Tensor oss = ori_sorted_src_.contiguous();
    const Tensor ogr = ori_group_rel_.contiguous();
    const Tensor ogd = ori_group_dst_.contiguous();
    const Tensor om = ori_mask_.contiguous().to(at::kFloat);
    const Tensor orsgp = ori_src_group_ptr_.contiguous();
    const Tensor olgs = ori_local_group_edge_start_.contiguous();
    const Tensor olgc = ori_local_group_edge_count_.contiguous();
    const Tensor olgr = ori_local_group_rel_.contiguous();
    const Tensor olgd = ori_local_group_dst_.contiguous();
    const Tensor olo = ori_local_order_.contiguous();
    const Tensor olsrc = ori_local_group_src_.contiguous();
    const Tensor oleg = ori_local_edge_group_.contiguous();
    const Tensor iss = inv_sorted_src_.contiguous();
    const Tensor igr = inv_group_rel_.contiguous();
    const Tensor igd = inv_group_dst_.contiguous();
    const Tensor im = inv_mask_.contiguous().to(at::kFloat);
    const Tensor irsgp = inv_src_group_ptr_.contiguous();
    const Tensor ilgs = inv_local_group_edge_start_.contiguous();
    const Tensor ilgc = inv_local_group_edge_count_.contiguous();
    const Tensor ilgr = inv_local_group_rel_.contiguous();
    const Tensor ilgd = inv_local_group_dst_.contiguous();
    const Tensor ilo = inv_local_order_.contiguous();
    const Tensor ilsrc = inv_local_group_src_.contiguous();
    const Tensor ileg = inv_local_edge_group_.contiguous();

    int64_t BL = A.size(0), E = A.size(1), n = w.size(1);
    int64_t ori_groups = ogr.size(0);
    int64_t inv_groups = igr.size(0);
    Tensor gA = at::zeros({BL, E}, A.options());
    Tensor gw = at::zeros({BL, n}, w.options());

    if (!wot_i) {
        int threads = 256;
        int blocks = (BL + threads - 1) / threads;
        fastlog_identity_backward_kernel<float><<<blocks, threads>>>(
            grad_ind.data_ptr<float>(), A.data_ptr<float>(), w.data_ptr<float>(),
            gA.data_ptr<float>(), gw.data_ptr<float>(), BL, E, n
        );
    }
    int threads = 256;
    if (!use_topk) {
        if (ori_groups > 0) {
            int64_t total = BL * ori_groups;
            int blocks = static_cast<int>((total + threads - 1) / threads);
            fastlog_backward_maxgroup_kernel<float><<<blocks, threads>>>(
                grad_ori.data_ptr<float>(), ori_arg.data_ptr<int32_t>(),
                A.data_ptr<float>(), w.data_ptr<float>(),
                oss.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(), ogd.data_ptr<int32_t>(), om.data_ptr<float>(),
                gA.data_ptr<float>(), gw.data_ptr<float>(),
                ori_groups, BL, E, n, 0
            );
        }
        if (inv_groups > 0) {
            int64_t total = BL * inv_groups;
            int blocks = static_cast<int>((total + threads - 1) / threads);
            fastlog_backward_maxgroup_kernel<float><<<blocks, threads>>>(
                grad_inv.data_ptr<float>(), inv_arg.data_ptr<int32_t>(),
                A.data_ptr<float>(), w.data_ptr<float>(),
                iss.data_ptr<int32_t>(), igr.data_ptr<int16_t>(), igd.data_ptr<int32_t>(), im.data_ptr<float>(),
                gA.data_ptr<float>(), gw.data_ptr<float>(),
                inv_groups, BL, E, n, r_size
            );
        }
    } else {
        auto launch_topk_bwd = [&](const Tensor &grad, const Tensor &arg, const Tensor &global_dst,
                                   const Tensor &lgr, const Tensor &lo, const Tensor &lsrc,
                                   const Tensor &leg, const Tensor &mask, int64_t offset) {
            int64_t total = BL * arg.size(1);
            int blocks = static_cast<int>((total + threads - 1) / threads);
            AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "fastlog_backward_maxgroup_topk_cuda", [&] {
                fastlog_backward_maxgroup_topk_kernel<scalar_t><<<blocks, threads>>>(
                    grad.data_ptr<scalar_t>(), arg.data_ptr<int32_t>(),
                    A.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
                    global_dst.data_ptr<int32_t>(),
                    lgr.data_ptr<int16_t>(), lo.data_ptr<int32_t>(), lsrc.data_ptr<int32_t>(), leg.data_ptr<int32_t>(), mask.data_ptr<scalar_t>(),
                    gA.data_ptr<scalar_t>(), gw.data_ptr<scalar_t>(),
                    arg.size(1), BL, E, n, offset
                );
            });
        };
        if (ori_groups > 0) {
            launch_topk_bwd(grad_ori, ori_arg, ogd, olgr, olo, olsrc, oleg, om, 0);
        }
        if (inv_groups > 0) {
            launch_topk_bwd(grad_inv, inv_arg, igd, ilgr, ilo, ilsrc, ileg, im, r_size);
        }
    }
    return std::make_tuple(gA, gw);
}

                                                                            
                                                                           
                                                                             
                                                                            
__global__ void mark_active_kernel(
    const float* __restrict__ A,
    int32_t* __restrict__ flags,
    int64_t BL, int64_t E
) {
    int64_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    for (int64_t bl = 0; bl < BL; bl++) {
        if (A[bl * E + e] != 0.0f) {
            flags[e] = 1;
            return;
        }
    }
}

Tensor compute_active_nodes_cuda(const Tensor &A_flat) {
    int64_t BL = A_flat.size(0), E = A_flat.size(1);
    auto flags = at::zeros({E}, A_flat.options().dtype(at::kInt));

    int threads = 256;
    int blocks = (E + threads - 1) / threads;
    mark_active_kernel<<<blocks, threads>>>(
        A_flat.data_ptr<float>(), flags.data_ptr<int32_t>(), BL, E
    );

    return flags.nonzero().reshape(-1).to(at::kLong);
}

                                                                            
                                                                          
                                                                     
                             
                                                                            
__global__ void apply_mask_kernel(
    const float* __restrict__ mask_values,
    const int32_t* __restrict__ order,
    const float* __restrict__ weight,                         
    float* __restrict__ out,
    int64_t nnz,
    bool has_weight
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;

    int64_t idx = order[i];
    float m = mask_values[idx];
    if (has_weight) {
        float w = weight[idx];
        float score = 1.0f / (1.0f + expf(-w));
        m *= score;
    }
    out[i] = m;
}

Tensor apply_mask_cuda(
    const Tensor &mask_values,
    const Tensor &order_,
    const Tensor &weight
) {
    const Tensor order = order_.contiguous();
    int64_t nnz = order.size(0);
    auto out = at::empty({nnz}, mask_values.options().dtype(at::kFloat));
    bool has_weight = weight.defined() && weight.numel() > 0;

    int threads = 256;
    int blocks = (nnz + threads - 1) / threads;
    apply_mask_kernel<<<blocks, threads>>>(
        mask_values.data_ptr<float>(),
        order.data_ptr<int32_t>(),
        has_weight ? weight.data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        nnz, has_weight
    );
    return out;
}

__device__ __forceinline__ void select_group_take_1d(
    int rel,
    int cnt,
    float score,
    int threshold_bucket,
    int64_t* remain,
    int* take_out
) {
    int take = cnt;
    if (threshold_bucket >= 0) {
        int bucket = bucketize_weight(score);
        if (bucket < threshold_bucket) {
            take = 0;
        } else if (bucket == threshold_bucket) {
            int64_t left = *remain;
            take = left > 0 ? static_cast<int>(left < cnt ? left : cnt) : 0;
            *remain -= take;
        }
    }
    *take_out = take;
}

__global__ void count_sparse3d_topk_kernel(
    const int64_t* __restrict__ sp_batch,
    const int64_t* __restrict__ sp_level,
    const int64_t* __restrict__ sp_entity,
    const float* __restrict__ w,
    const int32_t* __restrict__ row_group_ptr,
    const int16_t* __restrict__ group_rel,
    const int32_t* __restrict__ group_edge_count,
    int64_t* __restrict__ out_count,
    int64_t nnz,
    int64_t L,
    int64_t n,
    int64_t rel_offset,
    int64_t topk_edges
) {
    int64_t i = blockIdx.x;
    if (i >= nnz) return;
    int64_t b = sp_batch[i];
    int64_t l = sp_level[i];
    int64_t src = sp_entity[i];
    int64_t g0 = row_group_ptr[src], g1 = row_group_ptr[src + 1];

    __shared__ int counts[NUM_BUCKETS];
    __shared__ int64_t total;
    __shared__ int threshold_bucket;
    __shared__ int64_t remain;

    for (int bkt = threadIdx.x; bkt < NUM_BUCKETS; bkt += blockDim.x) {
        counts[bkt] = 0;
    }
    if (threadIdx.x == 0) {
        total = 0;
        threshold_bucket = -1;
        remain = 0;
    }
    __syncthreads();

    int64_t local_total = 0;
    for (int64_t g = g0 + threadIdx.x; g < g1; g += blockDim.x) {
        int rel = group_rel[g];
        int cnt = group_edge_count[g];
        int bucket = bucketize_weight(w[b * L * n + l * n + rel_offset + rel]);
        atomicAdd(&counts[bucket], cnt);
        local_total += cnt;
    }
    atomicAdd(reinterpret_cast<unsigned long long*>(&total), static_cast<unsigned long long>(local_total));
    __syncthreads();

    if (total <= topk_edges) {
        if (threadIdx.x == 0) out_count[i] = total;
        return;
    }

    if (threadIdx.x == 0) {
        threshold_bucket = NUM_BUCKETS - 1;
        remain = topk_edges;
        int64_t cumsum = 0;
        for (int bkt = NUM_BUCKETS - 1; bkt >= 0; --bkt) {
            int cnt = counts[bkt];
            if (cumsum + cnt >= topk_edges) {
                threshold_bucket = bkt;
                remain = topk_edges - cumsum;
                break;
            }
            cumsum += cnt;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int64_t selected = 0;
        int64_t remain_local = remain;
        for (int64_t g = g0; g < g1; ++g) {
            int rel = group_rel[g];
            int cnt = group_edge_count[g];
            int take = 0;
            select_group_take_1d(rel, cnt, w[b * L * n + l * n + rel_offset + rel], threshold_bucket, &remain_local, &take);
            selected += take;
        }
        out_count[i] = selected;
    }
}

template <typename scalar_t, bool IsInv>
__global__ void fill_sparse3d_topk_kernel(
    const int64_t* __restrict__ sp_batch,
    const int64_t* __restrict__ sp_level,
    const int64_t* __restrict__ sp_entity,
    const scalar_t* __restrict__ sp_value,
    const scalar_t* __restrict__ w,
    const int32_t* __restrict__ col_ind,
    const bool* __restrict__ mask_values,
    const int32_t* __restrict__ order,
    const scalar_t* __restrict__ edge_weight,
    bool has_edge_weight,
    bool edge_weight_is_score,
    double edge_weight_scale,
    const int32_t* __restrict__ row_group_ptr,
    const int16_t* __restrict__ group_rel,
    const int32_t* __restrict__ group_edge_start,
    const int32_t* __restrict__ group_edge_count,
    const int64_t* __restrict__ entry_offsets,
    int64_t* __restrict__ out_batch,
    int64_t* __restrict__ out_level,
    int64_t* __restrict__ out_entity,
    scalar_t* __restrict__ out_value,
    int64_t* __restrict__ meta_entry,
    int16_t* __restrict__ meta_rel,
    scalar_t* __restrict__ meta_mask,
    int64_t* __restrict__ meta_edge,
    int64_t nnz,
    int64_t L,
    int64_t n,
    int64_t rel_offset,
    int64_t topk_edges
) {
    int64_t i = blockIdx.x;
    if (i >= nnz) return;
    int64_t b = sp_batch[i];
    int64_t l = sp_level[i];
    int64_t src = sp_entity[i];
    scalar_t sv = sp_value[i];
    int64_t g0 = row_group_ptr[src], g1 = row_group_ptr[src + 1];

    __shared__ int counts[NUM_BUCKETS];
    __shared__ int threshold_bucket;
    __shared__ int64_t remain;
    __shared__ int64_t out_base;
    __shared__ int current_take;
    __shared__ int current_rel;
    __shared__ int32_t current_edge_start;

    for (int bkt = threadIdx.x; bkt < NUM_BUCKETS; bkt += blockDim.x) {
        counts[bkt] = 0;
    }
    if (threadIdx.x == 0) {
        threshold_bucket = -1;
        remain = 0;
        out_base = entry_offsets[i];
        current_take = 0;
        current_rel = 0;
        current_edge_start = 0;
    }
    __syncthreads();

    int64_t total = 0;
    for (int64_t g = g0 + threadIdx.x; g < g1; g += blockDim.x) {
        int rel = group_rel[g];
        int cnt = group_edge_count[g];
        int bucket = bucketize_weight(w[b * L * n + l * n + rel_offset + rel]);
        atomicAdd(&counts[bucket], cnt);
        total += cnt;
    }
    __shared__ int64_t total_shared;
    if (threadIdx.x == 0) total_shared = 0;
    __syncthreads();
    atomicAdd(reinterpret_cast<unsigned long long*>(&total_shared), static_cast<unsigned long long>(total));
    __syncthreads();

    if (threadIdx.x == 0 && total_shared > topk_edges) {
        threshold_bucket = NUM_BUCKETS - 1;
        int64_t cumsum = 0;
        for (int bkt = NUM_BUCKETS - 1; bkt >= 0; --bkt) {
            int cnt = counts[bkt];
            if (cumsum + cnt >= topk_edges) {
                threshold_bucket = bkt;
                remain = topk_edges - cumsum;
                break;
            }
            cumsum += cnt;
        }
    }
    __syncthreads();

    for (int64_t g = g0; g < g1; ++g) {
        if (threadIdx.x == 0) {
            current_rel = group_rel[g];
            int cnt = group_edge_count[g];
            current_take = cnt;
            if (threshold_bucket >= 0) {
                select_group_take_1d(current_rel, cnt, w[b * L * n + l * n + rel_offset + current_rel], threshold_bucket, &remain, &current_take);
            }
            current_edge_start = group_edge_start[g];
        }
        __syncthreads();

        for (int e = threadIdx.x; e < current_take; e += blockDim.x) {
            int64_t p = current_edge_start + e;
            int64_t out = out_base + e;
            int64_t edge_idx = static_cast<int64_t>(order[p]);
            scalar_t m = mask_values[edge_idx] ? scalar_t(1) : scalar_t(0);
            if (has_edge_weight) {
                scalar_t ew = edge_weight[edge_idx];
                scalar_t score = edge_weight_is_score ? ew : scalar_t(1) / (scalar_t(1) + expf(-ew * static_cast<scalar_t>(edge_weight_scale)));
                m *= score;
            }
            out_batch[out] = b;
            out_level[out] = l;
            out_entity[out] = col_ind[p];
            out_value[out] = sv * w[b * L * n + l * n + rel_offset + current_rel] * m;
            meta_entry[out] = i;
            meta_rel[out] = static_cast<int16_t>(current_rel);
            meta_mask[out] = m;
            meta_edge[out] = edge_idx;
        }
        __syncthreads();
        if (threadIdx.x == 0) out_base += current_take;
        __syncthreads();
    }
}

template <typename scalar_t, bool IsInv>
__global__ void fill_sparse3d_topk_masked_kernel(
    const int64_t* __restrict__ sp_batch,
    const int64_t* __restrict__ sp_level,
    const int64_t* __restrict__ sp_entity,
    const scalar_t* __restrict__ sp_value,
    const scalar_t* __restrict__ w,
    const int32_t* __restrict__ col_ind,
    const scalar_t* __restrict__ mask,
    const int32_t* __restrict__ row_group_ptr,
    const int16_t* __restrict__ group_rel,
    const int32_t* __restrict__ group_edge_start,
    const int32_t* __restrict__ group_edge_count,
    const int64_t* __restrict__ entry_offsets,
    int64_t* __restrict__ out_batch,
    int64_t* __restrict__ out_level,
    int64_t* __restrict__ out_entity,
    scalar_t* __restrict__ out_value,
    int64_t* __restrict__ meta_entry,
    int16_t* __restrict__ meta_rel,
    scalar_t* __restrict__ meta_mask,
    int64_t nnz,
    int64_t L,
    int64_t n,
    int64_t rel_offset,
    int64_t topk_edges
) {
    int64_t i = blockIdx.x;
    if (i >= nnz) return;
    int64_t b = sp_batch[i];
    int64_t l = sp_level[i];
    int64_t src = sp_entity[i];
    scalar_t sv = sp_value[i];
    int64_t g0 = row_group_ptr[src], g1 = row_group_ptr[src + 1];

    __shared__ int counts[NUM_BUCKETS];
    __shared__ int threshold_bucket;
    __shared__ int64_t remain;
    __shared__ int64_t out_base;
    __shared__ int current_take;
    __shared__ int current_rel;
    __shared__ int32_t current_edge_start;

    for (int bkt = threadIdx.x; bkt < NUM_BUCKETS; bkt += blockDim.x) {
        counts[bkt] = 0;
    }
    if (threadIdx.x == 0) {
        threshold_bucket = -1;
        remain = 0;
        out_base = entry_offsets[i];
        current_take = 0;
        current_rel = 0;
        current_edge_start = 0;
    }
    __syncthreads();

    int64_t total = 0;
    for (int64_t g = g0 + threadIdx.x; g < g1; g += blockDim.x) {
        int rel = group_rel[g];
        int cnt = group_edge_count[g];
        int bucket = bucketize_weight(w[b * L * n + l * n + rel_offset + rel]);
        atomicAdd(&counts[bucket], cnt);
        total += cnt;
    }
    __shared__ int64_t total_shared;
    if (threadIdx.x == 0) total_shared = 0;
    __syncthreads();
    atomicAdd(reinterpret_cast<unsigned long long*>(&total_shared), static_cast<unsigned long long>(total));
    __syncthreads();

    if (threadIdx.x == 0 && total_shared > topk_edges) {
        threshold_bucket = NUM_BUCKETS - 1;
        int64_t cumsum = 0;
        for (int bkt = NUM_BUCKETS - 1; bkt >= 0; --bkt) {
            int cnt = counts[bkt];
            if (cumsum + cnt >= topk_edges) {
                threshold_bucket = bkt;
                remain = topk_edges - cumsum;
                break;
            }
            cumsum += cnt;
        }
    }
    __syncthreads();

    for (int64_t g = g0; g < g1; ++g) {
        if (threadIdx.x == 0) {
            current_rel = group_rel[g];
            int cnt = group_edge_count[g];
            current_take = cnt;
            if (threshold_bucket >= 0) {
                select_group_take_1d(current_rel, cnt, w[b * L * n + l * n + rel_offset + current_rel], threshold_bucket, &remain, &current_take);
            }
            current_edge_start = group_edge_start[g];
        }
        __syncthreads();

        for (int e = threadIdx.x; e < current_take; e += blockDim.x) {
            int64_t p = current_edge_start + e;
            int64_t out = out_base + e;
            scalar_t m = mask[p];
            out_batch[out] = b;
            out_level[out] = l;
            out_entity[out] = col_ind[p];
            out_value[out] = sv * w[b * L * n + l * n + rel_offset + current_rel] * m;
            meta_entry[out] = i;
            meta_rel[out] = static_cast<int16_t>(current_rel);
            meta_mask[out] = m;
        }
        __syncthreads();
        if (threadIdx.x == 0) out_base += current_take;
        __syncthreads();
    }
}

__global__ void count_sparse3d_all_edges_kernel(
    const int64_t* __restrict__ sp_entity,
    const int32_t* __restrict__ row_group_ptr,
    const int32_t* __restrict__ group_edge_count,
    int64_t* __restrict__ out_count,
    int64_t nnz
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    int64_t src = sp_entity[i];
    int64_t g0 = row_group_ptr[src], g1 = row_group_ptr[src + 1];
    int64_t total = 0;
    for (int64_t g = g0; g < g1; ++g) total += group_edge_count[g];
    out_count[i] = total;
}

template <typename scalar_t, bool IsInv>
__global__ void fill_sparse3d_all_edges_kernel(
    const int64_t* __restrict__ sp_batch,
    const int64_t* __restrict__ sp_level,
    const int64_t* __restrict__ sp_entity,
    const scalar_t* __restrict__ sp_value,
    const scalar_t* __restrict__ w,
    const int32_t* __restrict__ col_ind,
    const scalar_t* __restrict__ mask,
    const int32_t* __restrict__ row_group_ptr,
    const int16_t* __restrict__ group_rel,
    const int32_t* __restrict__ group_edge_start,
    const int32_t* __restrict__ group_edge_count,
    const int64_t* __restrict__ entry_offsets,
    int64_t* __restrict__ out_batch,
    int64_t* __restrict__ out_level,
    int64_t* __restrict__ out_entity,
    scalar_t* __restrict__ out_value,
    int64_t* __restrict__ meta_entry,
    int16_t* __restrict__ meta_rel,
    scalar_t* __restrict__ meta_mask,
    int64_t nnz,
    int64_t L,
    int64_t n,
    int64_t rel_offset
) {
    int64_t i = blockIdx.x;
    if (i >= nnz) return;
    int64_t b = sp_batch[i];
    int64_t l = sp_level[i];
    int64_t src = sp_entity[i];
    scalar_t sv = sp_value[i];
    int64_t g0 = row_group_ptr[src], g1 = row_group_ptr[src + 1];
    __shared__ int current_rel;
    __shared__ int32_t current_edge_start;
    __shared__ int current_take;
    __shared__ int64_t out_base;
    if (threadIdx.x == 0) {
        current_rel = 0;
        current_edge_start = 0;
        current_take = 0;
        out_base = entry_offsets[i];
    }
    __syncthreads();

    for (int64_t g = g0; g < g1; ++g) {
        if (threadIdx.x == 0) {
            current_rel = group_rel[g];
            current_edge_start = group_edge_start[g];
            current_take = group_edge_count[g];
        }
        __syncthreads();
        for (int e = threadIdx.x; e < current_take; e += blockDim.x) {
            int64_t p = current_edge_start + e;
            int64_t out = out_base + e;
            scalar_t m = mask[p];
            out_batch[out] = b;
            out_level[out] = l;
            out_entity[out] = col_ind[p];
            out_value[out] = sv * w[b * L * n + l * n + rel_offset + current_rel] * m;
            meta_entry[out] = i;
            meta_rel[out] = static_cast<int16_t>(current_rel);
            meta_mask[out] = m;
        }
        __syncthreads();
        if (threadIdx.x == 0) out_base += current_take;
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void fill_sparse3d_identity_kernel(
    const int64_t* __restrict__ sp_batch,
    const int64_t* __restrict__ sp_level,
    const int64_t* __restrict__ sp_entity,
    const scalar_t* __restrict__ sp_value,
    const scalar_t* __restrict__ w,
    int64_t* __restrict__ out_batch,
    int64_t* __restrict__ out_level,
    int64_t* __restrict__ out_entity,
    scalar_t* __restrict__ out_value,
    int64_t nnz,
    int64_t L,
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    int64_t b = sp_batch[i];
    int64_t l = sp_level[i];
    out_batch[i] = b;
    out_level[i] = l;
    out_entity[i] = sp_entity[i];
    out_value[i] = sp_value[i] * w[b * L * n + l * n + (n - 1)];
}

template <typename scalar_t>
__global__ void backward_sparse3d_topk_entry_kernel(
    const scalar_t* __restrict__ grad_values,
    const int64_t* __restrict__ grad_keys,
    const int64_t* __restrict__ meta_entity,
    const int64_t* __restrict__ entry_offsets,
    const int16_t* __restrict__ meta_rel,
    const scalar_t* __restrict__ meta_mask,
    const int64_t* __restrict__ sp_batch,
    const int64_t* __restrict__ sp_level,
    const scalar_t* __restrict__ sp_value,
    const scalar_t* __restrict__ w,
    scalar_t* __restrict__ grad_sp_value,
    scalar_t* __restrict__ grad_w,
    int64_t nnz,
    int64_t grad_nnz,
    int64_t L,
    int64_t n,
    int64_t rel_offset,
    int64_t E
) {
    int64_t entry = blockIdx.x;
    if (entry >= nnz) return;

    int64_t b = sp_batch[entry];
    int64_t l = sp_level[entry];
    scalar_t sv = sp_value[entry];

    int64_t start = entry_offsets[entry];
    int64_t end = entry_offsets[entry + 1];
    scalar_t local_grad_sp = 0;

    for (int64_t k = start + threadIdx.x; k < end; k += blockDim.x) {
        int64_t key = b * (L * E) + l * E + meta_entity[k];
        int64_t lo = 0, hi = grad_nnz - 1, idx = -1;
        while (lo <= hi) {
            int64_t mid = (lo + hi) / 2;
            int64_t mid_key = grad_keys[mid];
            if (mid_key == key) { idx = mid; break; }
            if (mid_key < key) lo = mid + 1;
            else hi = mid - 1;
        }
        if (idx < 0) continue;

        int64_t rel = meta_rel[k];
        scalar_t m = meta_mask[k];
        scalar_t gv = grad_values[idx];
        scalar_t wv = w[b * L * n + l * n + rel_offset + rel];
        local_grad_sp += gv * m * wv;
        atomicAdd(&grad_w[b * L * n + l * n + rel_offset + rel], gv * m * sv);
    }

    if (local_grad_sp != 0) {
        atomicAdd(&grad_sp_value[entry], local_grad_sp);
    }
}

template <typename scalar_t>
__global__ void backward_sparse3d_identity_kernel(
    const scalar_t* __restrict__ grad_values,
    const int64_t* __restrict__ grad_keys,
    const int64_t* __restrict__ sp_batch,
    const int64_t* __restrict__ sp_level,
    const int64_t* __restrict__ sp_entity,
    const scalar_t* __restrict__ sp_value,
    const scalar_t* __restrict__ w,
    scalar_t* __restrict__ grad_sp_value,
    scalar_t* __restrict__ grad_w,
    int64_t nnz,
    int64_t grad_nnz,
    int64_t E,
    int64_t n,
    int64_t L
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    int64_t b = sp_batch[i];
    int64_t l = sp_level[i];
    int64_t e = sp_entity[i];
    int64_t key = b * (L * E) + l * E + e;
    int64_t lo = 0, hi = grad_nnz - 1, idx = -1;
    while (lo <= hi) {
        int64_t mid = (lo + hi) / 2;
        int64_t mid_key = grad_keys[mid];
        if (mid_key == key) { idx = mid; break; }
        if (mid_key < key) lo = mid + 1;
        else hi = mid - 1;
    }
    if (idx < 0) return;
    scalar_t gv = grad_values[idx];
    scalar_t wv = w[b * L * n + l * n + (n - 1)];
    atomicAdd(&grad_w[b * L * n + l * n + (n - 1)], gv * sp_value[i]);
    atomicAdd(&grad_sp_value[i], gv * wv);
}

template <typename scalar_t>
__global__ void backward_sparse3d_topk_aligned_kernel(
    const scalar_t* __restrict__ grad_raw_values,
    const int64_t* __restrict__ meta_entry,
    const int16_t* __restrict__ meta_rel,
    const scalar_t* __restrict__ meta_mask,
    const int64_t* __restrict__ sp_batch,
    const int64_t* __restrict__ sp_level,
    const scalar_t* __restrict__ sp_value,
    const scalar_t* __restrict__ w,
    scalar_t* __restrict__ grad_sp_value,
    scalar_t* __restrict__ grad_w,
    int64_t raw_nnz,
    int64_t L,
    int64_t n,
    int64_t rel_offset
) {
    int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= raw_nnz) return;
    scalar_t gv = grad_raw_values[k];
    if (gv == 0) return;
    int64_t entry = meta_entry[k];
    int64_t b = sp_batch[entry];
    int64_t l = sp_level[entry];
    int64_t rel = meta_rel[k];
    scalar_t m = meta_mask[k];
    scalar_t sv = sp_value[entry];
    scalar_t wv = w[b * L * n + l * n + rel_offset + rel];
    atomicAdd(&grad_sp_value[entry], gv * m * wv);
    atomicAdd(&grad_w[b * L * n + l * n + rel_offset + rel], gv * m * sv);
}

std::tuple<Tensor,Tensor,Tensor,Tensor, Tensor,Tensor,Tensor,Tensor, Tensor,Tensor,Tensor,Tensor,
           Tensor,Tensor,Tensor, Tensor,Tensor,Tensor, Tensor,Tensor, Tensor,Tensor>
fastlog_forward_sparse3d_topk_cuda(
    const Tensor &sp_batch_, const Tensor &sp_level_, const Tensor &sp_entity_, const Tensor &sp_value_,
    const Tensor &w_,
    const Tensor &ori_col_ind_, const Tensor &ori_order_,
    const Tensor &ori_row_group_ptr_, const Tensor &ori_group_rel_,
    const Tensor &ori_group_edge_start_, const Tensor &ori_group_edge_count_,
    const Tensor &inv_col_ind_, const Tensor &inv_order_,
    const Tensor &inv_row_group_ptr_, const Tensor &inv_group_rel_,
    const Tensor &inv_group_edge_start_, const Tensor &inv_group_edge_count_,
    const Tensor &mask_values_, const Tensor &edge_weight_,
    int64_t B, int64_t L, int64_t E, int64_t r_size, bool wot_i, int64_t topk_edges,
    bool edge_weight_is_score,
    double edge_weight_scale
) {
    const Tensor sp_batch = sp_batch_.contiguous();
    const Tensor sp_level = sp_level_.contiguous();
    const Tensor sp_entity = sp_entity_.contiguous();
    const Tensor sp_value = sp_value_.contiguous().to(at::kFloat);
    const Tensor w = w_.contiguous().to(at::kFloat);
    const Tensor oci = ori_col_ind_.contiguous();
    const Tensor oo = ori_order_.contiguous();
    const Tensor orgp = ori_row_group_ptr_.contiguous();
    const Tensor ogr = ori_group_rel_.contiguous();
    const Tensor ogs = ori_group_edge_start_.contiguous();
    const Tensor ogc = ori_group_edge_count_.contiguous();
    const Tensor ici = inv_col_ind_.contiguous();
    const Tensor io = inv_order_.contiguous();
    const Tensor irgp = inv_row_group_ptr_.contiguous();
    const Tensor igr = inv_group_rel_.contiguous();
    const Tensor igs = inv_group_edge_start_.contiguous();
    const Tensor igc = inv_group_edge_count_.contiguous();
    const Tensor mask_values = mask_values_.contiguous().to(at::kBool).view({-1});
    const bool has_edge_weight = edge_weight_.defined() && edge_weight_.numel() > 0;
    const Tensor edge_weight = has_edge_weight
        ? edge_weight_.contiguous().to(at::kFloat).view({-1})
        : Tensor();
    int64_t nnz = sp_batch.size(0);
    auto ori_count = at::zeros({nnz}, sp_batch.options().dtype(at::kLong));
    auto inv_count = at::zeros({nnz}, sp_batch.options().dtype(at::kLong));
    int threads_per_entry = 128;
    if (nnz > 0) {
        count_sparse3d_topk_kernel<<<nnz, threads_per_entry>>>(
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(),
            w.data_ptr<float>(), orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(), ogc.data_ptr<int32_t>(),
            ori_count.data_ptr<int64_t>(), nnz, L, w.size(2), 0, topk_edges
        );
        count_sparse3d_topk_kernel<<<nnz, threads_per_entry>>>(
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(),
            w.data_ptr<float>(), irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(), igc.data_ptr<int32_t>(),
            inv_count.data_ptr<int64_t>(), nnz, L, w.size(2), r_size, topk_edges
        );
    }
    auto ori_offsets = at::zeros({nnz + 1}, sp_batch.options().dtype(at::kLong));
    auto inv_offsets = at::zeros({nnz + 1}, sp_batch.options().dtype(at::kLong));
    if (nnz > 0) {
        ori_offsets.slice(0, 1, nnz + 1).copy_(ori_count.cumsum(0));
        inv_offsets.slice(0, 1, nnz + 1).copy_(inv_count.cumsum(0));
    }
    int64_t ori_total = ori_offsets[nnz].item<int64_t>();
    int64_t inv_total = inv_offsets[nnz].item<int64_t>();
    auto ori_b = at::empty({ori_total}, sp_batch.options().dtype(at::kLong));
    auto ori_l = at::empty({ori_total}, sp_batch.options().dtype(at::kLong));
    auto ori_e = at::empty({ori_total}, sp_batch.options().dtype(at::kLong));
    auto ori_v = at::empty({ori_total}, sp_value.options());
    auto ori_meta_entry = at::empty({ori_total}, sp_batch.options().dtype(at::kLong));
    auto ori_meta_rel = at::empty({ori_total}, sp_batch.options().dtype(at::kShort));
    auto ori_meta_mask = at::empty({ori_total}, sp_value.options());
    auto ori_meta_edge = at::empty({ori_total}, sp_batch.options().dtype(at::kLong));
    auto inv_b = at::empty({inv_total}, sp_batch.options().dtype(at::kLong));
    auto inv_l = at::empty({inv_total}, sp_batch.options().dtype(at::kLong));
    auto inv_e = at::empty({inv_total}, sp_batch.options().dtype(at::kLong));
    auto inv_v = at::empty({inv_total}, sp_value.options());
    auto inv_meta_entry = at::empty({inv_total}, sp_batch.options().dtype(at::kLong));
    auto inv_meta_rel = at::empty({inv_total}, sp_batch.options().dtype(at::kShort));
    auto inv_meta_mask = at::empty({inv_total}, sp_value.options());
    auto inv_meta_edge = at::empty({inv_total}, sp_batch.options().dtype(at::kLong));
    if (ori_total > 0) {
        fill_sparse3d_topk_kernel<float, false><<<nnz, threads_per_entry>>>(
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(), sp_value.data_ptr<float>(),
            w.data_ptr<float>(), oci.data_ptr<int32_t>(), mask_values.data_ptr<bool>(), oo.data_ptr<int32_t>(),
            has_edge_weight ? edge_weight.data_ptr<float>() : nullptr, has_edge_weight, edge_weight_is_score, edge_weight_scale,
            orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(), ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
            ori_offsets.data_ptr<int64_t>(),
            ori_b.data_ptr<int64_t>(), ori_l.data_ptr<int64_t>(), ori_e.data_ptr<int64_t>(), ori_v.data_ptr<float>(),
            ori_meta_entry.data_ptr<int64_t>(), ori_meta_rel.data_ptr<int16_t>(), ori_meta_mask.data_ptr<float>(),
            ori_meta_edge.data_ptr<int64_t>(),
            nnz, L, w.size(2), 0, topk_edges
        );
    }
    if (inv_total > 0) {
        fill_sparse3d_topk_kernel<float, true><<<nnz, threads_per_entry>>>(
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(), sp_value.data_ptr<float>(),
            w.data_ptr<float>(), ici.data_ptr<int32_t>(), mask_values.data_ptr<bool>(), io.data_ptr<int32_t>(),
            has_edge_weight ? edge_weight.data_ptr<float>() : nullptr, has_edge_weight, edge_weight_is_score, edge_weight_scale,
            irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(), igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
            inv_offsets.data_ptr<int64_t>(),
            inv_b.data_ptr<int64_t>(), inv_l.data_ptr<int64_t>(), inv_e.data_ptr<int64_t>(), inv_v.data_ptr<float>(),
            inv_meta_entry.data_ptr<int64_t>(), inv_meta_rel.data_ptr<int16_t>(), inv_meta_mask.data_ptr<float>(),
            inv_meta_edge.data_ptr<int64_t>(),
            nnz, L, w.size(2), r_size, topk_edges
        );
    }
    auto ind_b = at::empty({nnz}, sp_batch.options().dtype(at::kLong));
    auto ind_l = at::empty({nnz}, sp_batch.options().dtype(at::kLong));
    auto ind_e = at::empty({nnz}, sp_batch.options().dtype(at::kLong));
    auto ind_v = at::zeros({nnz}, sp_value.options());
    if (!wot_i && nnz > 0) {
        int threads = 256, blocks = (nnz + threads - 1) / threads;
        fill_sparse3d_identity_kernel<float><<<blocks, threads>>>(
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(),
            sp_value.data_ptr<float>(), w.data_ptr<float>(),
            ind_b.data_ptr<int64_t>(), ind_l.data_ptr<int64_t>(), ind_e.data_ptr<int64_t>(), ind_v.data_ptr<float>(),
            nnz, L, w.size(2)
        );
    } else if (wot_i) {
        ind_b.copy_(sp_batch); ind_l.copy_(sp_level); ind_e.copy_(sp_entity);
    }
    return std::make_tuple(
        ori_b, ori_l, ori_e, ori_v,
        inv_b, inv_l, inv_e, inv_v,
        ind_b, ind_l, ind_e, ind_v,
        ori_meta_entry, ori_meta_rel, ori_meta_mask,
        inv_meta_entry, inv_meta_rel, inv_meta_mask,
        ori_offsets, inv_offsets,
        ori_meta_edge, inv_meta_edge
    );
}

std::tuple<Tensor,Tensor,Tensor,Tensor, Tensor,Tensor,Tensor,Tensor, Tensor,Tensor,Tensor,Tensor,
           Tensor,Tensor,Tensor, Tensor,Tensor,Tensor, Tensor,Tensor>
fastlog_forward_sparse3d_topk_masked_cuda(
    const Tensor &sp_batch_, const Tensor &sp_level_, const Tensor &sp_entity_, const Tensor &sp_value_,
    const Tensor &w_,
    const Tensor &ori_col_ind_, const Tensor &ori_mask_,
    const Tensor &ori_row_group_ptr_, const Tensor &ori_group_rel_,
    const Tensor &ori_group_edge_start_, const Tensor &ori_group_edge_count_,
    const Tensor &inv_col_ind_, const Tensor &inv_mask_,
    const Tensor &inv_row_group_ptr_, const Tensor &inv_group_rel_,
    const Tensor &inv_group_edge_start_, const Tensor &inv_group_edge_count_,
    int64_t B, int64_t L, int64_t E, int64_t r_size, bool wot_i, int64_t topk_edges
) {
    const Tensor sp_batch = sp_batch_.contiguous();
    const Tensor sp_level = sp_level_.contiguous();
    const Tensor sp_entity = sp_entity_.contiguous();
    const Tensor sp_value = sp_value_.contiguous().to(at::kFloat);
    const Tensor w = w_.contiguous().to(at::kFloat);
    const Tensor oci = ori_col_ind_.contiguous();
    const Tensor om = ori_mask_.contiguous().to(at::kFloat);
    const Tensor orgp = ori_row_group_ptr_.contiguous();
    const Tensor ogr = ori_group_rel_.contiguous();
    const Tensor ogs = ori_group_edge_start_.contiguous();
    const Tensor ogc = ori_group_edge_count_.contiguous();
    const Tensor ici = inv_col_ind_.contiguous();
    const Tensor im = inv_mask_.contiguous().to(at::kFloat);
    const Tensor irgp = inv_row_group_ptr_.contiguous();
    const Tensor igr = inv_group_rel_.contiguous();
    const Tensor igs = inv_group_edge_start_.contiguous();
    const Tensor igc = inv_group_edge_count_.contiguous();
    int64_t nnz = sp_batch.size(0);
    auto ori_count = at::zeros({nnz}, sp_batch.options().dtype(at::kLong));
    auto inv_count = at::zeros({nnz}, sp_batch.options().dtype(at::kLong));
    int threads_per_entry = 128;
    if (nnz > 0) {
        count_sparse3d_topk_kernel<<<nnz, threads_per_entry>>>(
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(),
            w.data_ptr<float>(), orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(), ogc.data_ptr<int32_t>(),
            ori_count.data_ptr<int64_t>(), nnz, L, w.size(2), 0, topk_edges
        );
        count_sparse3d_topk_kernel<<<nnz, threads_per_entry>>>(
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(),
            w.data_ptr<float>(), irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(), igc.data_ptr<int32_t>(),
            inv_count.data_ptr<int64_t>(), nnz, L, w.size(2), r_size, topk_edges
        );
    }
    auto ori_offsets = at::zeros({nnz + 1}, sp_batch.options().dtype(at::kLong));
    auto inv_offsets = at::zeros({nnz + 1}, sp_batch.options().dtype(at::kLong));
    if (nnz > 0) {
        ori_offsets.slice(0, 1, nnz + 1).copy_(ori_count.cumsum(0));
        inv_offsets.slice(0, 1, nnz + 1).copy_(inv_count.cumsum(0));
    }
    int64_t ori_total = ori_offsets[nnz].item<int64_t>();
    int64_t inv_total = inv_offsets[nnz].item<int64_t>();
    auto ori_b = at::empty({ori_total}, sp_batch.options().dtype(at::kLong));
    auto ori_l = at::empty({ori_total}, sp_batch.options().dtype(at::kLong));
    auto ori_e = at::empty({ori_total}, sp_batch.options().dtype(at::kLong));
    auto ori_v = at::empty({ori_total}, sp_value.options());
    auto ori_meta_entry = at::empty({ori_total}, sp_batch.options().dtype(at::kLong));
    auto ori_meta_rel = at::empty({ori_total}, sp_batch.options().dtype(at::kShort));
    auto ori_meta_mask = at::empty({ori_total}, sp_value.options());
    auto inv_b = at::empty({inv_total}, sp_batch.options().dtype(at::kLong));
    auto inv_l = at::empty({inv_total}, sp_batch.options().dtype(at::kLong));
    auto inv_e = at::empty({inv_total}, sp_batch.options().dtype(at::kLong));
    auto inv_v = at::empty({inv_total}, sp_value.options());
    auto inv_meta_entry = at::empty({inv_total}, sp_batch.options().dtype(at::kLong));
    auto inv_meta_rel = at::empty({inv_total}, sp_batch.options().dtype(at::kShort));
    auto inv_meta_mask = at::empty({inv_total}, sp_value.options());
    if (ori_total > 0) {
        fill_sparse3d_topk_masked_kernel<float, false><<<nnz, threads_per_entry>>>(
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(), sp_value.data_ptr<float>(),
            w.data_ptr<float>(), oci.data_ptr<int32_t>(), om.data_ptr<float>(),
            orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(), ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
            ori_offsets.data_ptr<int64_t>(),
            ori_b.data_ptr<int64_t>(), ori_l.data_ptr<int64_t>(), ori_e.data_ptr<int64_t>(), ori_v.data_ptr<float>(),
            ori_meta_entry.data_ptr<int64_t>(), ori_meta_rel.data_ptr<int16_t>(), ori_meta_mask.data_ptr<float>(),
            nnz, L, w.size(2), 0, topk_edges
        );
    }
    if (inv_total > 0) {
        fill_sparse3d_topk_masked_kernel<float, true><<<nnz, threads_per_entry>>>(
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(), sp_value.data_ptr<float>(),
            w.data_ptr<float>(), ici.data_ptr<int32_t>(), im.data_ptr<float>(),
            irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(), igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
            inv_offsets.data_ptr<int64_t>(),
            inv_b.data_ptr<int64_t>(), inv_l.data_ptr<int64_t>(), inv_e.data_ptr<int64_t>(), inv_v.data_ptr<float>(),
            inv_meta_entry.data_ptr<int64_t>(), inv_meta_rel.data_ptr<int16_t>(), inv_meta_mask.data_ptr<float>(),
            nnz, L, w.size(2), r_size, topk_edges
        );
    }
    auto ind_b = at::empty({nnz}, sp_batch.options().dtype(at::kLong));
    auto ind_l = at::empty({nnz}, sp_batch.options().dtype(at::kLong));
    auto ind_e = at::empty({nnz}, sp_batch.options().dtype(at::kLong));
    auto ind_v = at::zeros({nnz}, sp_value.options());
    if (!wot_i && nnz > 0) {
        int threads = 256, blocks = (nnz + threads - 1) / threads;
        fill_sparse3d_identity_kernel<float><<<blocks, threads>>>(
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(),
            sp_value.data_ptr<float>(), w.data_ptr<float>(),
            ind_b.data_ptr<int64_t>(), ind_l.data_ptr<int64_t>(), ind_e.data_ptr<int64_t>(), ind_v.data_ptr<float>(),
            nnz, L, w.size(2)
        );
    } else if (wot_i) {
        ind_b.copy_(sp_batch); ind_l.copy_(sp_level); ind_e.copy_(sp_entity);
    }
    return std::make_tuple(
        ori_b, ori_l, ori_e, ori_v,
        inv_b, inv_l, inv_e, inv_v,
        ind_b, ind_l, ind_e, ind_v,
        ori_meta_entry, ori_meta_rel, ori_meta_mask,
        inv_meta_entry, inv_meta_rel, inv_meta_mask,
        ori_offsets, inv_offsets
    );
}

std::tuple<Tensor,Tensor,Tensor,Tensor, Tensor,Tensor,Tensor,Tensor, Tensor,Tensor,Tensor,Tensor,
           Tensor,Tensor,Tensor, Tensor,Tensor,Tensor, Tensor,Tensor>
fastlog_forward_sparse3d_alledges_cuda(
    const Tensor &sp_batch_, const Tensor &sp_level_, const Tensor &sp_entity_, const Tensor &sp_value_,
    const Tensor &w_,
    const Tensor &ori_col_ind_, const Tensor &ori_mask_,
    const Tensor &ori_row_group_ptr_, const Tensor &ori_group_rel_,
    const Tensor &ori_group_edge_start_, const Tensor &ori_group_edge_count_,
    const Tensor &inv_col_ind_, const Tensor &inv_mask_,
    const Tensor &inv_row_group_ptr_, const Tensor &inv_group_rel_,
    const Tensor &inv_group_edge_start_, const Tensor &inv_group_edge_count_,
    int64_t B, int64_t L, int64_t E, int64_t r_size, bool wot_i
) {
    const Tensor sp_batch = sp_batch_.contiguous();
    const Tensor sp_level = sp_level_.contiguous();
    const Tensor sp_entity = sp_entity_.contiguous();
    const Tensor sp_value = sp_value_.contiguous().to(at::kFloat);
    const Tensor w = w_.contiguous().to(at::kFloat);
    const Tensor oci = ori_col_ind_.contiguous();
    const Tensor om = ori_mask_.contiguous().to(at::kFloat);
    const Tensor orgp = ori_row_group_ptr_.contiguous();
    const Tensor ogr = ori_group_rel_.contiguous();
    const Tensor ogs = ori_group_edge_start_.contiguous();
    const Tensor ogc = ori_group_edge_count_.contiguous();
    const Tensor ici = inv_col_ind_.contiguous();
    const Tensor im = inv_mask_.contiguous().to(at::kFloat);
    const Tensor irgp = inv_row_group_ptr_.contiguous();
    const Tensor igr = inv_group_rel_.contiguous();
    const Tensor igs = inv_group_edge_start_.contiguous();
    const Tensor igc = inv_group_edge_count_.contiguous();
    int64_t nnz = sp_batch.size(0);
    auto ori_count = at::zeros({nnz}, sp_batch.options().dtype(at::kLong));
    auto inv_count = at::zeros({nnz}, sp_batch.options().dtype(at::kLong));
    int threads_per_entry = 128;
    if (nnz > 0) {
        int blocks = static_cast<int>((nnz + 255) / 256);
        count_sparse3d_all_edges_kernel<<<blocks, 256>>>(
            sp_entity.data_ptr<int64_t>(), orgp.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
            ori_count.data_ptr<int64_t>(), nnz
        );
        count_sparse3d_all_edges_kernel<<<blocks, 256>>>(
            sp_entity.data_ptr<int64_t>(), irgp.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
            inv_count.data_ptr<int64_t>(), nnz
        );
    }
    auto ori_offsets = at::zeros({nnz + 1}, sp_batch.options().dtype(at::kLong));
    auto inv_offsets = at::zeros({nnz + 1}, sp_batch.options().dtype(at::kLong));
    if (nnz > 0) {
        ori_offsets.slice(0, 1, nnz + 1).copy_(ori_count.cumsum(0));
        inv_offsets.slice(0, 1, nnz + 1).copy_(inv_count.cumsum(0));
    }
    int64_t ori_total = ori_offsets[nnz].item<int64_t>();
    int64_t inv_total = inv_offsets[nnz].item<int64_t>();
    auto ori_b = at::empty({ori_total}, sp_batch.options().dtype(at::kLong));
    auto ori_l = at::empty({ori_total}, sp_batch.options().dtype(at::kLong));
    auto ori_e = at::empty({ori_total}, sp_batch.options().dtype(at::kLong));
    auto ori_v = at::empty({ori_total}, sp_value.options());
    auto ori_meta_entry = at::empty({ori_total}, sp_batch.options().dtype(at::kLong));
    auto ori_meta_rel = at::empty({ori_total}, sp_batch.options().dtype(at::kShort));
    auto ori_meta_mask = at::empty({ori_total}, sp_value.options());
    auto inv_b = at::empty({inv_total}, sp_batch.options().dtype(at::kLong));
    auto inv_l = at::empty({inv_total}, sp_batch.options().dtype(at::kLong));
    auto inv_e = at::empty({inv_total}, sp_batch.options().dtype(at::kLong));
    auto inv_v = at::empty({inv_total}, sp_value.options());
    auto inv_meta_entry = at::empty({inv_total}, sp_batch.options().dtype(at::kLong));
    auto inv_meta_rel = at::empty({inv_total}, sp_batch.options().dtype(at::kShort));
    auto inv_meta_mask = at::empty({inv_total}, sp_value.options());
    if (ori_total > 0) {
        fill_sparse3d_all_edges_kernel<float, false><<<nnz, threads_per_entry>>>(
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(), sp_value.data_ptr<float>(),
            w.data_ptr<float>(), oci.data_ptr<int32_t>(), om.data_ptr<float>(),
            orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(), ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
            ori_offsets.data_ptr<int64_t>(),
            ori_b.data_ptr<int64_t>(), ori_l.data_ptr<int64_t>(), ori_e.data_ptr<int64_t>(), ori_v.data_ptr<float>(),
            ori_meta_entry.data_ptr<int64_t>(), ori_meta_rel.data_ptr<int16_t>(), ori_meta_mask.data_ptr<float>(),
            nnz, L, w.size(2), 0
        );
    }
    if (inv_total > 0) {
        fill_sparse3d_all_edges_kernel<float, true><<<nnz, threads_per_entry>>>(
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(), sp_value.data_ptr<float>(),
            w.data_ptr<float>(), ici.data_ptr<int32_t>(), im.data_ptr<float>(),
            irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(), igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
            inv_offsets.data_ptr<int64_t>(),
            inv_b.data_ptr<int64_t>(), inv_l.data_ptr<int64_t>(), inv_e.data_ptr<int64_t>(), inv_v.data_ptr<float>(),
            inv_meta_entry.data_ptr<int64_t>(), inv_meta_rel.data_ptr<int16_t>(), inv_meta_mask.data_ptr<float>(),
            nnz, L, w.size(2), r_size
        );
    }
    auto ind_b = at::empty({nnz}, sp_batch.options().dtype(at::kLong));
    auto ind_l = at::empty({nnz}, sp_batch.options().dtype(at::kLong));
    auto ind_e = at::empty({nnz}, sp_batch.options().dtype(at::kLong));
    auto ind_v = at::zeros({nnz}, sp_value.options());
    if (!wot_i && nnz > 0) {
        int threads = 256, blocks = (nnz + threads - 1) / threads;
        fill_sparse3d_identity_kernel<float><<<blocks, threads>>>(
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(),
            sp_value.data_ptr<float>(), w.data_ptr<float>(),
            ind_b.data_ptr<int64_t>(), ind_l.data_ptr<int64_t>(), ind_e.data_ptr<int64_t>(), ind_v.data_ptr<float>(),
            nnz, L, w.size(2)
        );
    } else if (wot_i) {
        ind_b.copy_(sp_batch); ind_l.copy_(sp_level); ind_e.copy_(sp_entity);
    }
    return std::make_tuple(
        ori_b, ori_l, ori_e, ori_v,
        inv_b, inv_l, inv_e, inv_v,
        ind_b, ind_l, ind_e, ind_v,
        ori_meta_entry, ori_meta_rel, ori_meta_mask,
        inv_meta_entry, inv_meta_rel, inv_meta_mask,
        ori_offsets, inv_offsets
    );
}

std::tuple<Tensor, Tensor>
fastlog_backward_sparse3d_topk_cuda(
    const Tensor &grad_ori_values_, const Tensor &grad_ori_batch_, const Tensor &grad_ori_level_, const Tensor &grad_ori_entity_,
    const Tensor &grad_inv_values_, const Tensor &grad_inv_batch_, const Tensor &grad_inv_level_, const Tensor &grad_inv_entity_,
    const Tensor &grad_ind_values_, const Tensor &grad_ind_batch_, const Tensor &grad_ind_level_, const Tensor &grad_ind_entity_,
    const Tensor &sp_batch_, const Tensor &sp_level_, const Tensor &sp_entity_, const Tensor &sp_value_, const Tensor &w_,
    const Tensor &ori_meta_batch_, const Tensor &ori_meta_level_, const Tensor &ori_meta_entity_,
    const Tensor &ori_meta_entry_, const Tensor &ori_meta_rel_, const Tensor &ori_meta_mask_, const Tensor &ori_offsets_,
    const Tensor &inv_meta_batch_, const Tensor &inv_meta_level_, const Tensor &inv_meta_entity_,
    const Tensor &inv_meta_entry_, const Tensor &inv_meta_rel_, const Tensor &inv_meta_mask_, const Tensor &inv_offsets_,
    int64_t B, int64_t L, int64_t E, int64_t r_size, bool wot_i
) {
    const Tensor sp_batch = sp_batch_.contiguous();
    const Tensor sp_level = sp_level_.contiguous();
    const Tensor sp_entity = sp_entity_.contiguous();
    const Tensor sp_value = sp_value_.contiguous().to(at::kFloat);
    const Tensor w = w_.contiguous().to(at::kFloat);
    int64_t nnz = sp_batch.size(0);
    int64_t n = w.size(2);
    auto grad_sp_value = at::zeros({nnz}, sp_value.options());
    auto grad_w = at::zeros({B, L, n}, w.options());
    const Tensor ori_meta_entity = ori_meta_entity_.contiguous();
    const Tensor ori_offsets = ori_offsets_.contiguous();
    if (grad_ori_batch_.size(0) > 0 && ori_meta_entry_.size(0) > 0) {
        auto gb = grad_ori_batch_.contiguous();
        auto gl = grad_ori_level_.contiguous();
        auto ge = grad_ori_entity_.contiguous();
        auto gv = grad_ori_values_.contiguous().to(at::kFloat);
        auto gkeys = gb * (L * E) + gl * E + ge;
        int threads = 128, blocks = static_cast<int>(nnz);
        backward_sparse3d_topk_entry_kernel<float><<<blocks, threads>>>(
            gv.data_ptr<float>(), gkeys.data_ptr<int64_t>(),
            ori_meta_entity.data_ptr<int64_t>(),
            ori_offsets.data_ptr<int64_t>(), ori_meta_rel_.data_ptr<int16_t>(), ori_meta_mask_.data_ptr<float>(),
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_value.data_ptr<float>(), w.data_ptr<float>(),
            grad_sp_value.data_ptr<float>(), grad_w.data_ptr<float>(),
            nnz, gb.size(0), L, n, 0, E
        );
    }
    const Tensor inv_meta_entity = inv_meta_entity_.contiguous();
    const Tensor inv_offsets = inv_offsets_.contiguous();
    if (grad_inv_batch_.size(0) > 0 && inv_meta_entry_.size(0) > 0) {
        auto gb = grad_inv_batch_.contiguous();
        auto gl = grad_inv_level_.contiguous();
        auto ge = grad_inv_entity_.contiguous();
        auto gv = grad_inv_values_.contiguous().to(at::kFloat);
        auto gkeys = gb * (L * E) + gl * E + ge;
        int threads = 128, blocks = static_cast<int>(nnz);
        backward_sparse3d_topk_entry_kernel<float><<<blocks, threads>>>(
            gv.data_ptr<float>(), gkeys.data_ptr<int64_t>(),
            inv_meta_entity.data_ptr<int64_t>(),
            inv_offsets.data_ptr<int64_t>(), inv_meta_rel_.data_ptr<int16_t>(), inv_meta_mask_.data_ptr<float>(),
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_value.data_ptr<float>(), w.data_ptr<float>(),
            grad_sp_value.data_ptr<float>(), grad_w.data_ptr<float>(),
            nnz, gb.size(0), L, n, r_size, E
        );
    }
    if (!wot_i && grad_ind_batch_.size(0) > 0) {
        auto gb = grad_ind_batch_.contiguous();
        auto gl = grad_ind_level_.contiguous();
        auto ge = grad_ind_entity_.contiguous();
        auto gv = grad_ind_values_.contiguous().to(at::kFloat);
        auto gkeys = gb * (L * E) + gl * E + ge;
        int threads = 256, blocks = (nnz + threads - 1) / threads;
        backward_sparse3d_identity_kernel<float><<<blocks, threads>>>(
            gv.data_ptr<float>(), gkeys.data_ptr<int64_t>(),
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(),
            sp_value.data_ptr<float>(), w.data_ptr<float>(),
            grad_sp_value.data_ptr<float>(), grad_w.data_ptr<float>(), nnz, gv.size(0), E, n, L
        );
    }
    return std::make_tuple(grad_sp_value, grad_w);
}

std::tuple<Tensor, Tensor>
fastlog_backward_sparse3d_topk_aligned_cuda(
    const Tensor &grad_ori_raw_values_,
    const Tensor &ori_meta_entry_, const Tensor &ori_meta_rel_, const Tensor &ori_meta_mask_,
    const Tensor &grad_inv_raw_values_,
    const Tensor &inv_meta_entry_, const Tensor &inv_meta_rel_, const Tensor &inv_meta_mask_,
    const Tensor &grad_ind_values_, const Tensor &grad_ind_batch_, const Tensor &grad_ind_level_, const Tensor &grad_ind_entity_,
    const Tensor &sp_batch_, const Tensor &sp_level_, const Tensor &sp_entity_, const Tensor &sp_value_, const Tensor &w_,
    int64_t B, int64_t L, int64_t E, int64_t r_size, bool wot_i
) {
    const Tensor sp_batch = sp_batch_.contiguous();
    const Tensor sp_level = sp_level_.contiguous();
    const Tensor sp_entity = sp_entity_.contiguous();
    const Tensor sp_value = sp_value_.contiguous().to(at::kFloat);
    const Tensor w = w_.contiguous().to(at::kFloat);
    int64_t nnz = sp_batch.size(0);
    int64_t n = w.size(2);
    auto grad_sp_value = at::zeros({nnz}, sp_value.options());
    auto grad_w = at::zeros({B, L, n}, w.options());

    int threads = 256;
    if (grad_ori_raw_values_.numel() > 0) {
        auto gv = grad_ori_raw_values_.contiguous().to(at::kFloat);
        int blocks = (gv.size(0) + threads - 1) / threads;
        backward_sparse3d_topk_aligned_kernel<float><<<blocks, threads>>>(
            gv.data_ptr<float>(),
            ori_meta_entry_.data_ptr<int64_t>(),
            ori_meta_rel_.data_ptr<int16_t>(),
            ori_meta_mask_.data_ptr<float>(),
            sp_batch.data_ptr<int64_t>(),
            sp_level.data_ptr<int64_t>(),
            sp_value.data_ptr<float>(),
            w.data_ptr<float>(),
            grad_sp_value.data_ptr<float>(),
            grad_w.data_ptr<float>(),
            gv.size(0), L, n, 0
        );
    }
    if (grad_inv_raw_values_.numel() > 0) {
        auto gv = grad_inv_raw_values_.contiguous().to(at::kFloat);
        int blocks = (gv.size(0) + threads - 1) / threads;
        backward_sparse3d_topk_aligned_kernel<float><<<blocks, threads>>>(
            gv.data_ptr<float>(),
            inv_meta_entry_.data_ptr<int64_t>(),
            inv_meta_rel_.data_ptr<int16_t>(),
            inv_meta_mask_.data_ptr<float>(),
            sp_batch.data_ptr<int64_t>(),
            sp_level.data_ptr<int64_t>(),
            sp_value.data_ptr<float>(),
            w.data_ptr<float>(),
            grad_sp_value.data_ptr<float>(),
            grad_w.data_ptr<float>(),
            gv.size(0), L, n, r_size
        );
    }
    if (!wot_i && grad_ind_batch_.size(0) > 0) {
        auto gb = grad_ind_batch_.contiguous();
        auto gl = grad_ind_level_.contiguous();
        auto ge = grad_ind_entity_.contiguous();
        auto gv = grad_ind_values_.contiguous().to(at::kFloat);
        auto gkeys = gb * (L * E) + gl * E + ge;
        int blocks = (nnz + threads - 1) / threads;
        backward_sparse3d_identity_kernel<float><<<blocks, threads>>>(
            gv.data_ptr<float>(), gkeys.data_ptr<int64_t>(),
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(),
            sp_value.data_ptr<float>(), w.data_ptr<float>(),
            grad_sp_value.data_ptr<float>(), grad_w.data_ptr<float>(), nnz, gv.size(0), E, n, L
        );
    }
    return std::make_tuple(grad_sp_value, grad_w);
}

}                     
