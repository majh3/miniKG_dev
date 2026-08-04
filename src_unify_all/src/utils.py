import time
import numpy as np
import torch
import torch_scatter
from torch.nn import functional as F

import os
import json

from logger import info

class Option(object):
    def __init__(self, d, exp_sig):
        self.__dict__ = d
        self.exp_dir = os.path.join(self.exps_dir, self.exp_name, exp_sig)
                                               
                                     
        if os.path.exists(self.exp_dir):
            self.exp_dir = self.exp_dir + "_" + time.strftime("%H-%M-%S")
        os.makedirs(self.exp_dir)
    def save(self):
        with open(os.path.join(self.exp_dir, "option.txt"), "w") as f:
            json.dump(self.__dict__, f, indent=1)
        return True
    
def activation(x, one):
    return torch.minimum(x, one.to(x.device))

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_sum(src, index, dim=-1, dim_size=None):
    index = broadcast(index, src, dim)
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)

def scatter_max(src, index, dim=-1, dim_size=None):
    index = broadcast(index, src, dim)
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return torch_scatter.scatter_max(src, index, dim, out, dim_size)[0]

def create_compact(class_tensor, value_tensor, target_size):
    max_vals = scatter_max(value_tensor, class_tensor, dim=-1, dim_size=target_size)
                                                                         
    return max_vals

def create_type(index_tensor):
    batch_size = index_tensor.shape[0]
    L = index_tensor.shape[1]
    K = index_tensor.shape[2]
    index_tensor = index_tensor.view(-1, K)
    index_new = []
    type_set = []
    max_size = 0
    for i in range(batch_size * L):
        types = torch.unique(index_tensor[i])
        type_index = torch.searchsorted(types, index_tensor[i])
        index_new.append(type_index)
        type_set.append(types)
        if types.shape[0] > max_size:
            max_size = types.shape[0]
    index_new = torch.stack(index_new, dim=0).view(batch_size, L, -1)

    type_set_pad = torch.zeros(batch_size * L, max_size + 1).long().to(index_tensor.device)
    for i in range(batch_size * L):
        types = type_set[i]
        type_set_pad[i][:types.shape[0]] = types
    
    return index_new, type_set_pad.view(batch_size, L, -1)

def create_type_sp(index_tensor):
    index_tensor = index_tensor.t()
    L = index_tensor.shape[0]
    K = index_tensor.shape[1]
    index_new = []
    type_set = []
    max_size = 0
    for i in range(L):
        types = torch.unique(index_tensor[i])
        type_index = torch.searchsorted(types, index_tensor[i])
        index_new.append(type_index)
        type_set.append(types)
        if types.shape[0] > max_size:
            max_size = types.shape[0]
    index_new = torch.stack(index_new, dim=0).view(L, -1)

    type_set_pad = torch.zeros(L, max_size + 1).long().to(index_tensor.device)
    for i in range(L):
        types = type_set[i]
        type_set_pad[i][:types.shape[0]] = types
    
    return index_new, type_set_pad

def block_is_in(arr, targets, block_size=10000000):
    if arr.numel() == 0 or targets.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=arr.device)

    result_list = []
    num_blocks = (arr.size(0) + block_size - 1) // block_size

    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, arr.size(0))
        block = arr[start_idx:end_idx]
        block_result = torch.isin(block, targets)
        block_result = torch.nonzero(block_result)[:, 0]
        result_list.append(block_result + start_idx)

    if len(result_list) == 0:
        return torch.empty(0, dtype=torch.long, device=arr.device)
    return torch.cat(result_list)

def log_loss(p_score, label, E, tau_2, one, thr=1e-7):
    if p_score.layout != torch.strided:
        p_score = p_score.to_dense()
    one_hot = F.one_hot(label, E).float()
    loss = -torch.sum(
        one_hot * torch.log(torch.maximum(p_score / tau_2, one * thr)),
        dim=-1)
    loss = torch.mean(loss)
    return loss

def log_loss_sm(p_score, label, E, tau_2, one, thr=1e-7):
    one_hot = F.one_hot(label, E).float()
    loss = -torch.sum(one_hot * torch.log(torch.maximum(torch.softmax(p_score / tau_2, -1), one * thr)), dim=-1)
    loss = torch.mean(loss)
    return loss

def log_loss_focal(p_score, label, E, tau_2, one, thr=1e-7):
    one_hot = F.one_hot(label, E).float()
    neg_one_hot = 1 - one_hot
    loss = -torch.sum(
        one_hot * torch.log(torch.maximum(p_score / tau_2, one * thr)),
        dim=-1)
    loss = torch.mean(loss)
    return loss

def log_loss_focal_probs(p_score, label, E, tau_2, one, thr=1e-7, plane=None):
    one_hot = F.one_hot(label, E).float()
    probs = torch.softmax(p_score, dim=-1)
    if plane is not None:
        pos_weights = one_hot*p_score
        pos_weights = (1 - pos_weights)**2
        neg_weights = (1 - one_hot)*p_score
        neg_weights = (neg_weights)**2
        weights = pos_weights + neg_weights
    else:
        weights = torch.ones_like(p_score)
    loss = -torch.sum(
        weights * one_hot * torch.log(torch.maximum(probs/tau_2, one * thr)) + \
        weights * (1 - one_hot) * torch.log(torch.maximum((1 - probs)/tau_2, one * thr)),
        dim=-1)
                                                                                                         
    loss = torch.mean(loss)
    return loss

def log_loss_neg(p_score, label, E, tau_2, one, thr=1e-7, neg_size=10):
                                           
    answer_score = torch.gather(p_score, dim=-1, index=label.unsqueeze(-1))
    batch_size = label.shape[0]
    neg_index = torch.randint(0, int(E), [batch_size, neg_size]).to(answer_score.device)
    neg_score = torch.gather(p_score, index=neg_index, dim=-1)
    logits = torch.cat([answer_score, neg_score], dim=-1)
    target = torch.zeros_like(logits)
    target[:, 0] = 1
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    weights = torch.ones_like(logits)
    with torch.no_grad():
        weights[:, 1:] = F.softmax(logits[:, 1:]/ tau_2, dim=-1)

    loss = (loss * weights.unsqueeze(1)).mean()
    return loss

def log_loss_common(p_score, label, E, tau_2, thr=1e-7):
                                           
    i_y = label.long()
    i_x = torch.arange(0, i_y.shape[0]).to(i_y.device)
    i = torch.stack([i_x, i_y], dim=0)
    v = torch.ones_like(label).float()
    one_hot = torch.sparse_coo_tensor(i.long(), v, torch.Size([i_y.shape[0], E])).to_dense()
    loss = -torch.sum(
        one_hot * torch.log(torch.maximum(p_score / tau_2, torch.ones_like(p_score) * thr)),
        dim=-1)
    loss = torch.mean(loss)
    return loss

def log_loss_common_sp(p_score, label, E, tau_2, thr=1e-7):
    if getattr(p_score, "is_sparse", False):
        p_score = p_score.coalesce()
        idx = p_score.indices()
        val = p_score.values()
        i_y = label.long()
        target_key = torch.arange(0, i_y.shape[0], device=i_y.device, dtype=idx.dtype) * int(E) + i_y.to(idx.dtype)
        key = idx[0] * int(E) + idx[1]
        if key.numel() == 0:
            logits = val.new_zeros(i_y.shape[0])
            loss = -torch.mean(
                torch.log(torch.maximum(logits / tau_2, torch.ones_like(logits) * thr)),
                dim=-1)
            return loss
        pos = torch.searchsorted(key, target_key)
        valid = (pos < key.numel()) & (key[pos.clamp_max(key.numel() - 1)] == target_key)
        logits = val.new_zeros(i_y.shape[0])
        if bool(valid.any().item()):
            valid_idx = torch.nonzero(valid, as_tuple=False).view(-1)
            logits = logits.scatter(0, valid_idx, val[pos[valid_idx]])
        loss = -torch.mean(
            torch.log(torch.maximum(logits / tau_2, torch.ones_like(logits) * thr)),
            dim=-1)
        return loss

    i_y = label.long()
    i_x = torch.arange(0, i_y.shape[0]).to(i_y.device)
    i = torch.stack([i_x, i_y], dim=0)
    v = torch.ones_like(label).float()
    one_hot = torch.sparse_coo_tensor(i.long(), v, torch.Size([i_y.shape[0], E]))
    logits = torch.sparse.sum(p_score * one_hot, dim=-1).to_dense()
    loss = -torch.mean(
        torch.log(torch.maximum(logits / tau_2, torch.ones_like(logits) * thr)),
        dim=-1)
    return loss

def sparse_matrix_multiply(A, B, target_size, r_size, tau_1, is_training=False, dropout=None,
                           is_max=False, weight=None, wot_i=False):

    scatter = scatter_sum
    if is_max: scatter = scatter_max

    row_indices, col_indices, r_indices, mask_values, w = B

    non_zero = torch.nonzero(A) 
    non_zero = torch.unique(non_zero[:, 1]) 

    non_zero_ori = block_is_in(row_indices, non_zero)                           
    row_indices_ori = torch.index_select(row_indices, index=non_zero_ori, dim=0)                    
    col_indices_ori = torch.index_select(col_indices, index=non_zero_ori, dim=0)  
    mask_values_ori = torch.index_select(mask_values, index=non_zero_ori, dim=0) 
    r_indices_ori = torch.index_select(r_indices, index=non_zero_ori, dim=0) 
                               
    if weight is not None:
        C_ori = torch.index_select(weight, index=non_zero_ori.to(weight.device), dim=0).to(A.device)
        C_ori = score_function_2(C_ori)  

    non_zero_inv = block_is_in(col_indices, non_zero)                      
    row_indices_inv = torch.index_select(col_indices, index=non_zero_inv, dim=0)              
    col_indices_inv = torch.index_select(row_indices, index=non_zero_inv, dim=0)              
    mask_values_inv = torch.index_select(mask_values, index=non_zero_inv, dim=0)
    r_indices_inv = torch.index_select(r_indices, index=non_zero_inv, dim=0)
    if weight is not None:
        C_inv = torch.index_select(weight, index=non_zero_inv.to(weight.device), dim=0).to(A.device)
        C_inv = score_function_2(C_inv)

    w = torch.softmax(w / tau_1, dim=-1)

    A_values_ori = torch.index_select(A, dim=1, index=row_indices_ori)                        
    B_values_ori = torch.index_select(w[:, :, :r_size], index=r_indices_ori, dim=2)                           
    result_values_ori = torch.einsum('bm,blm->blm', A_values_ori, B_values_ori) * mask_values_ori.unsqueeze(0).unsqueeze(0)
    if weight is not None: result_values_ori = result_values_ori * C_ori.t().unsqueeze(dim=0)
    result_ori = scatter(result_values_ori, col_indices_ori.long(), dim=2, dim_size=target_size)                      

    A_values_inv = torch.index_select(A, dim=1, index=row_indices_inv)                                                                                        
    B_values_inv = torch.index_select(w[:, :, r_size:2 * r_size], index=r_indices_inv, dim=2)                      
    result_values_inv = torch.einsum('bm,blm->blm', A_values_inv, B_values_inv) * mask_values_inv.unsqueeze(0).unsqueeze(0)                                                        
                                                                                                    
    
    if weight is not None: result_values_inv = result_values_inv * C_inv.t().unsqueeze(dim=0)                  
    result_inv = scatter(result_values_inv, col_indices_inv.long(), dim=2, dim_size=target_size)                      

    result_ind = None
    if not wot_i:  
                                
        result_ind = torch.einsum('bm,bl->blm', A, w[:, :, -1])                      
    
    return result_ind, result_ori, result_inv

def sparse_matrix_multiply_L_sample(A, B, target_size, r_size, tau_1, is_training=False,
                                    dropout=None, is_max=False, top_k=1000, topk_pruning=100000, weight=None,
                                    use_topk=False, wot_i=False, inverse=False):
    scatter = scatter_sum
    if is_max: scatter = scatter_max

    row_indices, col_indices, r_indices, mask_values, w = B

    non_zero = torch.nonzero(A.sum(1))
    non_zero = torch.unique(non_zero[:, 1])
    if use_topk:
        k_ = min(top_k, A.shape[-1])
        topk = torch.topk(A.sum(1), k=k_)[1]
        topk = torch.unique(topk.view(-1))
        if topk.shape[0] < non_zero.shape[0]:
            non_zero = topk

    non_zero_ori = block_is_in(row_indices, non_zero)
    row_indices_ori = torch.index_select(row_indices, index=non_zero_ori, dim=0)
    col_indices_ori = torch.index_select(col_indices, index=non_zero_ori, dim=0)
    mask_values_ori = torch.index_select(mask_values, index=non_zero_ori, dim=0)
    r_indices_ori = torch.index_select(r_indices, index=non_zero_ori, dim=0)
    if weight is not None:
        C_ori = torch.index_select(weight, index=non_zero_ori.to(weight.device), dim=0).to(A.device)
        C_ori = score_function_2(C_ori)

    non_zero_inv = block_is_in(col_indices, non_zero)
    row_indices_inv = torch.index_select(col_indices, index=non_zero_inv, dim=0)
    col_indices_inv = torch.index_select(row_indices, index=non_zero_inv, dim=0)
    mask_values_inv = torch.index_select(mask_values, index=non_zero_inv, dim=0)
    r_indices_inv = torch.index_select(r_indices, index=non_zero_inv, dim=0)
    if weight is not None:
        C_inv = torch.index_select(weight, index=non_zero_inv.to(weight.device), dim=0).to(A.device)
        C_inv = score_function_2(C_inv)

    w = torch.softmax(w / tau_1, dim=-1)

    A_values_ori = torch.index_select(A, dim=-1, index=row_indices_ori)                       
    if use_topk:
        k_ = min(topk_pruning, A_values_ori.shape[-1])
        A_values_ori_topk, A_values_ori_topk_indices = torch.topk(A_values_ori, k=k_)
        B_values_ori = torch.index_select(w[:, :, :r_size], index=r_indices_ori, dim=2)
        B_values_ori_topk = torch.gather(B_values_ori, index=A_values_ori_topk_indices, dim=-1)
        mask_values_ori_topk = mask_values_ori[A_values_ori_topk_indices]
        result_values_ori = A_values_ori_topk * B_values_ori_topk * mask_values_ori_topk
        if weight is not None:
            C_ori_topk = C_ori.squeeze(dim=-1)[A_values_ori_topk_indices]
            result_values_ori = result_values_ori * C_ori_topk
                                                                        
        col_indices_ori = col_indices_ori[A_values_ori_topk_indices]
        result_ori = scatter(result_values_ori, col_indices_ori.long(), dim=2, dim_size=target_size)
    else:
        B_values_ori = torch.index_select(w[:, :, :r_size], index=r_indices_ori, dim=2)             
        result_values_ori = A_values_ori * B_values_ori * mask_values_ori
                                                                                                        
        if weight is not None:
            result_values_ori = result_values_ori * C_ori.squeeze(dim=-1)
                                                                        
        result_ori = scatter(result_values_ori, col_indices_ori.long(), dim=2, dim_size=target_size)

    A_values_inv = torch.index_select(A, dim=-1, index=row_indices_inv)
    if use_topk:
        k_ = min(topk_pruning, A_values_inv.shape[-1])
        A_values_inv_topk, A_values_inv_topk_indices = torch.topk(A_values_inv, k=k_)
        B_values_inv = torch.index_select(w[:, :, r_size:2 * r_size], index=r_indices_inv, dim=2)
        B_values_inv_topk = torch.gather(B_values_inv, index=A_values_inv_topk_indices, dim=-1)
        mask_values_inv_topk = mask_values_inv[A_values_inv_topk_indices]
        result_values_inv = A_values_inv_topk * B_values_inv_topk * mask_values_inv_topk
        if weight is not None:
            C_inv_topk = C_inv.squeeze(dim=-1)[A_values_inv_topk_indices]
            result_values_inv = result_values_inv * C_inv_topk
                                                                        
        col_indices_inv = col_indices_inv[A_values_inv_topk_indices]
        result_inv = scatter(result_values_inv, col_indices_inv.long(), dim=2, dim_size=target_size)
    else:
        B_values_inv = torch.index_select(w[:, :, r_size:2 * r_size], index=r_indices_inv, dim=2)
        result_values_inv = A_values_inv * B_values_inv * mask_values_inv
                                                                                                        
        if weight is not None:
            result_values_inv = result_values_inv * C_inv.squeeze(dim=-1)
                                                                        
        result_inv = scatter(result_values_inv, col_indices_inv.long(), dim=2, dim_size=target_size)

    result_ind = None
    if not wot_i and not inverse:
        result_ind = torch.einsum('ble,bl->ble', A, w[:, :, -1])

    return result_ind, result_ori, result_inv

def sparse_matrix_multiply_max(A, B, target_size, r_size, tau_1, is_training=False, dropout=None,
                           is_max=False, weight=None, wot_i=False):

    row_indices, col_indices, r_indices, mask_values, w = B

    non_zero = torch.nonzero(A)
    non_zero = torch.unique(non_zero[:, 1])
    non_zero_ori = block_is_in(row_indices, non_zero)
    row_indices_ori = torch.index_select(row_indices, index=non_zero_ori, dim=0)
    col_indices_ori = torch.index_select(col_indices, index=non_zero_ori, dim=0)
    mask_values_ori = torch.index_select(mask_values, index=non_zero_ori, dim=0)
    r_indices_ori = torch.index_select(r_indices, index=non_zero_ori, dim=0)
    if weight is not None:
        C_ori = torch.index_select(weight, index=non_zero_ori.to(weight.device), dim=0).to(A.device)
        C_ori = score_function_2(C_ori)

    non_zero_inv = block_is_in(col_indices, non_zero)
    row_indices_inv = torch.index_select(col_indices, index=non_zero_inv, dim=0)
    col_indices_inv = torch.index_select(row_indices, index=non_zero_inv, dim=0)
    mask_values_inv = torch.index_select(mask_values, index=non_zero_inv, dim=0)
    r_indices_inv = torch.index_select(r_indices, index=non_zero_inv, dim=0)
    if weight is not None:
        C_inv = torch.index_select(weight, index=non_zero_inv.to(weight.device), dim=0).to(A.device)
        C_inv = score_function_2(C_inv)

    w = torch.softmax(w / tau_1, dim=-1)

    A_values_ori = torch.index_select(A.detach(), dim=1, index=row_indices_ori)
    B_values_ori = torch.index_select(w[:, :, :r_size], index=r_indices_ori, dim=2)
    result_values_ori = torch.einsum('bm,blm->blm', A_values_ori, B_values_ori) * mask_values_ori.unsqueeze(0).unsqueeze(0)
    if weight is not None: result_values_ori = result_values_ori * C_ori.t().unsqueeze(dim=0)
                                                                    
    index_ori = col_indices_ori.long() * r_size + r_indices_ori.long()
    type_ori = torch.unique(index_ori)
    type_index_ori = torch.searchsorted(type_ori, index_ori)
    result_values_ori = create_compact(type_index_ori, result_values_ori, type_ori.shape[0])
    col_indices_ori = type_ori // r_size
    result_ori = scatter_sum(result_values_ori, col_indices_ori, dim=2, dim_size=target_size)

    A_values_inv = torch.index_select(A.detach(), dim=1, index=row_indices_inv)
    B_values_inv = torch.index_select(w[:, :, r_size:2 * r_size], index=r_indices_inv, dim=2)
    result_values_inv = torch.einsum('bm,blm->blm', A_values_inv, B_values_inv) * mask_values_inv.unsqueeze(0).unsqueeze(0)
    if weight is not None: result_values_inv = result_values_inv * C_inv.t().unsqueeze(dim=0)
                                                                    
    index_inv = col_indices_inv.long() * r_size + r_indices_inv.long()
    type_inv = torch.unique(index_inv)
    type_index_inv = torch.searchsorted(type_inv, index_inv)
    result_values_inv = create_compact(type_index_inv, result_values_inv, type_inv.shape[0])
    col_indices_inv = type_inv // r_size
    result_inv = scatter_sum(result_values_inv, col_indices_inv, dim=2, dim_size=target_size)
    result_ind = None
    if not wot_i:
        result_ind = torch.einsum('bm,bl->blm', A.detach(), w[:, :, -1])
    return result_ind, result_ori, result_inv

def sparse_matrix_multiply_L_sample_max(A, B, target_size, r_size, tau_1, is_training=False,
                                    dropout=None, is_max=False, top_k=1000, topk_pruning=100000, weight=None,
                                    use_topk=False, wot_i=False):
    scatter = scatter_sum
    if is_max: scatter = scatter_max

    row_indices, col_indices, r_indices, mask_values, w = B

    non_zero = torch.nonzero(A.sum(1))
    non_zero = torch.unique(non_zero[:, 1])
    if use_topk:
        k_ = min(top_k, A.shape[-1])
        topk = torch.topk(A.sum(1), k=k_)[1]
        topk = torch.unique(topk.view(-1))
        if topk.shape[0] < non_zero.shape[0]:
            non_zero = topk

    non_zero_ori = block_is_in(row_indices, non_zero)
    row_indices_ori = torch.index_select(row_indices, index=non_zero_ori, dim=0)
    col_indices_ori = torch.index_select(col_indices, index=non_zero_ori, dim=0)
    mask_values_ori = torch.index_select(mask_values, index=non_zero_ori, dim=0)
    r_indices_ori = torch.index_select(r_indices, index=non_zero_ori, dim=0)
    if weight is not None:
        C_ori = torch.index_select(weight, index=non_zero_ori.to(weight.device), dim=0).to(A.device)
        C_ori = score_function_2(C_ori)

    non_zero_inv = block_is_in(col_indices, non_zero)
    row_indices_inv = torch.index_select(col_indices, index=non_zero_inv, dim=0)
    col_indices_inv = torch.index_select(row_indices, index=non_zero_inv, dim=0)
    mask_values_inv = torch.index_select(mask_values, index=non_zero_inv, dim=0)
    r_indices_inv = torch.index_select(r_indices, index=non_zero_inv, dim=0)
    if weight is not None:
        C_inv = torch.index_select(weight, index=non_zero_inv.to(weight.device), dim=0).to(A.device)
        C_inv = score_function_2(C_inv)

    w = torch.softmax(w / tau_1, dim=-1)

    A_values_ori = torch.index_select(A, dim=-1, index=row_indices_ori)
    if use_topk:
        k_ = min(topk_pruning, A_values_ori.shape[-1])
        A_values_ori_topk, A_values_ori_topk_indices = torch.topk(A_values_ori, k=k_)
        B_values_ori = torch.index_select(w[:, :, :r_size], index=r_indices_ori, dim=2)
        B_values_ori_topk = torch.gather(B_values_ori, index=A_values_ori_topk_indices, dim=-1)
        mask_values_ori_topk = mask_values_ori[A_values_ori_topk_indices]
        result_values_ori = A_values_ori_topk * B_values_ori_topk * mask_values_ori_topk
        if weight is not None:
            C_ori_topk = C_ori.squeeze(dim=-1)[A_values_ori_topk_indices]
            result_values_ori = result_values_ori * C_ori_topk
                                                                        
        col_indices_ori = col_indices_ori[A_values_ori_topk_indices]
        index_ori = col_indices_ori.long() * r_size + r_indices_ori[A_values_ori_topk_indices].long()
        type_index_ori, type_ori = create_type(index_ori)
        result_values_ori = create_compact(type_index_ori, result_values_ori, type_ori.shape[-1])
        col_indices_ori = type_ori // r_size
        result_ori = scatter_sum(result_values_ori, col_indices_ori, dim=2, dim_size=target_size)
    else:
        B_values_ori = torch.index_select(w[:, :, :r_size], index=r_indices_ori, dim=2)
        result_values_ori = A_values_ori * B_values_ori * mask_values_ori
        if weight is not None:
            result_values_ori = result_values_ori * C_ori.squeeze(dim=-1)
                                                                        
        index_ori = col_indices_ori.long() * r_size + r_indices_ori.long()
        type_ori = torch.unique(index_ori)
        type_index_ori = torch.searchsorted(type_ori, index_ori)
        result_values_ori = create_compact(type_index_ori, result_values_ori, type_ori.shape[0])
        col_indices_ori = type_ori // r_size
        result_ori = scatter_sum(result_values_ori, col_indices_ori, dim=2, dim_size=target_size)

    A_values_inv = torch.index_select(A, dim=-1, index=row_indices_inv)
    if use_topk:
        k_ = min(topk_pruning, A_values_inv.shape[-1])
        A_values_inv_topk, A_values_inv_topk_indices = torch.topk(A_values_inv, k=k_)
        B_values_inv = torch.index_select(w[:, :, r_size:2 * r_size], index=r_indices_inv, dim=2)
        B_values_inv_topk = torch.gather(B_values_inv, index=A_values_inv_topk_indices, dim=-1)
        mask_values_inv_topk = mask_values_inv[A_values_inv_topk_indices]
        result_values_inv = A_values_inv_topk * B_values_inv_topk * mask_values_inv_topk
        if weight is not None:
            C_inv_topk = C_inv.squeeze(dim=-1)[A_values_inv_topk_indices]
            result_values_inv = result_values_inv * C_inv_topk
                                                                        
        col_indices_inv = col_indices_inv[A_values_inv_topk_indices]
        index_inv = col_indices_inv.long() * r_size + r_indices_inv[A_values_inv_topk_indices].long()
        type_index_inv, type_inv = create_type(index_inv)
        result_values_inv = create_compact(type_index_inv, result_values_inv, type_inv.shape[-1])
        col_indices_inv = type_inv // r_size
        result_inv = scatter_sum(result_values_inv, col_indices_inv, dim=2, dim_size=target_size)
    else:
        B_values_inv = torch.index_select(w[:, :, r_size:2 * r_size], index=r_indices_inv, dim=2)
        result_values_inv = A_values_inv * B_values_inv * mask_values_inv
        if weight is not None:
            result_values_inv = result_values_inv * C_inv.squeeze(dim=-1)
                                                                        
        index_inv = col_indices_inv.long() * r_size + r_indices_inv.long()
        type_inv = torch.unique(index_inv)
        type_index_inv = torch.searchsorted(type_inv, index_inv)
        result_values_inv = create_compact(type_index_inv, result_values_inv, type_inv.shape[0])
        col_indices_inv = type_inv // r_size
        result_inv = scatter_sum(result_values_inv, col_indices_inv, dim=2, dim_size=target_size)

    result_ind = None
    if not wot_i:
        result_ind = torch.einsum('ble,bl->ble', A, w[:, :, -1])

    return result_ind, result_ori, result_inv

def sparse_matrix_multiply_sp(A, B, E, r_size, tau_1, is_training=False, dropout=None, is_max=False,
                              wot_i=False, weight=None):
    scatter = scatter_sum
    if is_max: scatter = scatter_max

    row_indices_ori, col_indices_ori, r_indices_ori, mask_values_ori, w_all = B
    indices_all, results_all = [], []
    indices_all_inv, results_all_inv = [], []
    indices_all_ind, results_all_ind = [], []
    batch_size = A.shape[0]
    L = w_all.shape[1]
    for i in range(batch_size):
        non_zero_ori = A[i]
        w = w_all[i].t()
        non_zero = block_is_in(row_indices_ori, non_zero_ori)
        non_zero_inv = block_is_in(col_indices_ori, non_zero_ori)

        col_indices = torch.index_select(col_indices_ori, index=non_zero, dim=0)
        mask_values = torch.index_select(mask_values_ori, index=non_zero, dim=0)
        r_indices = torch.index_select(r_indices_ori, index=non_zero, dim=0)
        if weight is not None:
            C_ori = torch.index_select(weight, index=non_zero.to(weight.device), dim=0).to(A.device)
            C_ori = score_function_2(C_ori)

        col_indices_inv = torch.index_select(row_indices_ori, index=non_zero_inv, dim=0)
        mask_values_inv = torch.index_select(mask_values_ori, index=non_zero_inv, dim=0)
        r_indices_inv = torch.index_select(r_indices_ori, index=non_zero_inv, dim=0)
        if weight is not None:
            C_inv = torch.index_select(weight, index=non_zero_inv.to(weight.device), dim=0).to(A.device)
            C_inv = score_function_2(C_inv)

        w = torch.softmax(w / tau_1, dim=0)

        B_values = torch.index_select(w[:r_size, :], index=r_indices, dim=0)
        result_values = B_values.t() * mask_values.unsqueeze(0)
        if weight is not None: result_values = result_values * C_ori.t()
                                                                     
        col_indices_uni = torch.unique(col_indices)
        sorted_indices = torch.searchsorted(col_indices_uni, col_indices)
        result = scatter(result_values, sorted_indices.long(), dim=1, dim_size=col_indices_uni.shape[0])
                                                 
                                                                        
        index = torch.ones_like(col_indices_uni) * i
        index = torch.stack([index, col_indices_uni], dim=0)
        indices_all.append(index)
        results_all.append(result)

                                                                    
        B_values_inv = torch.index_select(w[r_size: 2 * r_size, :], index=r_indices_inv, dim=0)
        result_values_inv = B_values_inv.t() * mask_values_inv.unsqueeze(0)
        if weight is not None: result_values_inv = result_values_inv * C_inv.t()
                                                                             
        col_indices_uni_inv = torch.unique(col_indices_inv)
        sorted_indices_inv = torch.searchsorted(col_indices_uni_inv, col_indices_inv)
        result_inv = scatter(result_values_inv, sorted_indices_inv.long(), dim=1, dim_size=col_indices_uni_inv.shape[0])
                                                     
                                                                                
        index = torch.ones_like(col_indices_uni_inv) * i
        index = torch.stack([index, col_indices_uni_inv], dim=0)
        indices_all_inv.append(index)
        results_all_inv.append(result_inv)

        if not wot_i:
            index = torch.ones_like(A[i]) * i
            index = torch.stack([index, A[i]], dim=0)
            indices_all_ind.append(index.unsqueeze(dim=-1))
            results_all_ind.append(w[-1:, :])
    i = torch.cat(indices_all, dim=-1)
    v = torch.cat(results_all, dim=-1).t()
    output_ori = torch.sparse_coo_tensor(i.long(), v, torch.Size([batch_size, E, L]))

    i = torch.cat(indices_all_inv, dim=-1)
    v = torch.cat(results_all_inv, dim=-1).t()
    output_inv = torch.sparse_coo_tensor(i.long(), v, torch.Size([batch_size, E, L]))

    if not wot_i:
        i = torch.cat(indices_all_ind, dim=-1)
        v = torch.cat(results_all_ind, dim=0)
        output_ind = torch.sparse_coo_tensor(i.long(), v, torch.Size([batch_size, E, L]))
    else:
        output_ind = None

    return output_ind, output_ori, output_inv

def sparse_matrix_multiply_L_sp(A, B, E, r_size, tau_1, is_training=False, dropout=None, is_max=False,
                                top_k=1000, topk_pruning=100000, wot_i=False, weight=None, use_topk=False):
    scatter = scatter_sum
    if is_max: scatter = scatter_max

    row_indices_ori, col_indices_ori, r_indices_ori, mask_values_ori, w_all = B
    indices_all, results_all = [], []
    indices_all_inv, results_all_inv = [], []
    indices_all_ind, results_all_ind = [], []
    A_batch = torch.unbind(A, dim=0)
    batch_size = len(A_batch)
    L = w_all.shape[1]
    for i in range(batch_size):
        w = w_all[i].t()
        A_ = A_batch[i].coalesce()
        A_indices = A_.indices()[0]
        A_values = A_.values()
        non_zero_ori = A_indices
        if use_topk:
            k_ = min(top_k, A_values.shape[0])
            topk = torch.topk(A_values.sum(1), k=k_)[1]
            if topk.shape[0] < non_zero_ori.shape[0]:
                non_zero_ori = A_indices[topk]
        non_zero = block_is_in(row_indices_ori, non_zero_ori)
        row_indices = torch.index_select(row_indices_ori, index=non_zero, dim=0)
        col_indices = torch.index_select(col_indices_ori, index=non_zero, dim=0)
        mask_values = torch.index_select(mask_values_ori, index=non_zero, dim=0)
        r_indices = torch.index_select(r_indices_ori, index=non_zero, dim=0)
        if weight is not None:
            C_ori = torch.index_select(weight, index=non_zero.to(weight.device), dim=0).to(A.device)
            C_ori = score_function_2(C_ori)

        non_zero_inv = block_is_in(col_indices_ori, non_zero_ori)
        row_indices_inv = torch.index_select(col_indices_ori, index=non_zero_inv, dim=0)
        col_indices_inv = torch.index_select(row_indices_ori, index=non_zero_inv, dim=0)
        mask_values_inv = torch.index_select(mask_values_ori, index=non_zero_inv, dim=0)
        r_indices_inv = torch.index_select(r_indices_ori, index=non_zero_inv, dim=0)
        if weight is not None:
            C_inv = torch.index_select(weight, index=non_zero_inv.to(weight.device), dim=0).to(A.device)
            C_inv = score_function_2(C_inv)

        w = torch.softmax(w / tau_1, dim=0)

        sorted_indices = torch.searchsorted(A_indices, row_indices)
        A_values_ori = torch.index_select(A_values, dim=0, index=sorted_indices)
        if use_topk:
            k_ = min(topk_pruning, A_values_ori.shape[0])
            A_values_ori_topk, A_values_ori_topk_indices = torch.topk(A_values_ori, k=k_, dim=0)
            B_values_ori = torch.index_select(w[:r_size, :], index=r_indices, dim=0)
            B_values_ori_topk = torch.gather(B_values_ori, index=A_values_ori_topk_indices, dim=0)
            mask_values_ori_topk = mask_values[A_values_ori_topk_indices]
            result_values_ori = A_values_ori_topk * B_values_ori_topk * mask_values_ori_topk
            if weight is not None:
                C_ori_topk = C_ori.squeeze(dim=1)[A_values_ori_topk_indices]
                result_values_ori = result_values_ori * C_ori_topk
                                                                            
            col_indices_ori_topk = col_indices[A_values_ori_topk_indices]
            col_indices_uni = torch.unique(col_indices_ori_topk)
            sorted_indices = torch.searchsorted(col_indices_uni, col_indices_ori_topk)
            result_ori = scatter(result_values_ori, sorted_indices, dim=0, dim_size=col_indices_uni.shape[0])
        else:
            B_values_ori = torch.index_select(w[:r_size, :], index=r_indices, dim=0)
            result_values_ori = A_values_ori * B_values_ori * mask_values.unsqueeze(dim=1)
            if weight is not None:
                result_values_ori = result_values_ori * C_ori.squeeze(dim=1)
                                                                            
            col_indices_uni = torch.unique(col_indices)
            sorted_indices = torch.searchsorted(col_indices_uni, col_indices)
            result_ori = scatter(result_values_ori, sorted_indices, dim=0, dim_size=col_indices_uni.shape[0])

                                                 
                                                                  
        index = torch.ones_like(col_indices_uni) * i
        index = torch.stack([index, col_indices_uni], dim=0)
        indices_all.append(index)
        results_all.append(result_ori)

        sorted_indices_inv = torch.searchsorted(A_indices, row_indices_inv)
        A_values_inv = torch.index_select(A_values, dim=0, index=sorted_indices_inv)
        if use_topk:
            k_ = min(topk_pruning, A_values_inv.shape[0])
            A_values_inv_topk, A_values_inv_topk_indices = torch.topk(A_values_inv, k=k_, dim=0)
            B_values_inv = torch.index_select(w[r_size: 2 * r_size, :], index=r_indices_inv, dim=0)
            B_values_inv_topk = torch.gather(B_values_inv, index=A_values_inv_topk_indices, dim=0)
            mask_values_inv_topk = mask_values_inv[A_values_inv_topk_indices]
            result_values_inv = A_values_inv_topk * B_values_inv_topk * mask_values_inv_topk
            if weight is not None:
                C_inv_topk = C_inv.squeeze(dim=1)[A_values_inv_topk_indices]
                result_values_inv = result_values_inv * C_inv_topk
                                                                            
            col_indices_inv_topk = col_indices_inv[A_values_inv_topk_indices]
            col_indices_uni_inv = torch.unique(col_indices_inv_topk)
            sorted_indices_inv = torch.searchsorted(col_indices_uni_inv, col_indices_inv_topk)
            result_inv = scatter(result_values_inv, sorted_indices_inv.long(), dim=0,
                                     dim_size=col_indices_uni_inv.shape[0])
        else:
            B_values_inv = torch.index_select(w[r_size: 2 * r_size, :], index=r_indices_inv, dim=0)
            result_values_inv = A_values_inv * B_values_inv * mask_values_inv.unsqueeze(dim=1)
            if weight is not None:
                result_values_inv = result_values_inv * C_inv.squeeze(dim=1)
                                                                            
            col_indices_uni_inv = torch.unique(col_indices_inv)
            sorted_indices_inv = torch.searchsorted(col_indices_uni_inv, col_indices_inv)
            result_inv = scatter(result_values_inv, sorted_indices_inv, dim=0,
                                 dim_size=col_indices_uni_inv.shape[0])

        index = torch.ones_like(col_indices_uni_inv) * i
        index = torch.stack([index, col_indices_uni_inv], dim=0)
        indices_all_inv.append(index)
        results_all_inv.append(result_inv)

        if not wot_i:
            result_ind = A_values * w[-1, :].unsqueeze(dim=0)
            index = torch.ones_like(A_indices) * i
            index = torch.stack([index, A_indices], dim=0)
            indices_all_ind.append(index)
            results_all_ind.append(result_ind)

    i = torch.cat(indices_all, dim=-1)
    v = torch.cat(results_all, dim=0)
    output_ori = torch.sparse_coo_tensor(i.long(), v, torch.Size([batch_size, E, L]))

    i = torch.cat(indices_all_inv, dim=-1)
    v = torch.cat(results_all_inv, dim=0)
    output_inv = torch.sparse_coo_tensor(i.long(), v, torch.Size([batch_size, E, L]))

    if not wot_i:
        i = torch.cat(indices_all_ind, dim=-1)
        v = torch.cat(results_all_ind, dim=0)
        output_ind = torch.sparse_coo_tensor(i.long(), v, torch.Size([batch_size, E, L]))
    else:
        output_ind = None

    return output_ind, output_ori, output_inv

def sparse_matrix_multiply_sp_max(A, B, E, r_size, tau_1, is_training=False, dropout=None, is_max=False,
                              wot_i=False, weight=None):
    scatter = scatter_sum

    row_indices_ori, col_indices_ori, r_indices_ori, mask_values_ori, w_all = B
    indices_all, results_all = [], []
    indices_all_inv, results_all_inv = [], []
    indices_all_ind, results_all_ind = [], []
    batch_size = A.shape[0]
    L = w_all.shape[1]
    for i in range(batch_size):
        non_zero_ori = A[i]
        w = w_all[i].t()
        non_zero = block_is_in(row_indices_ori, non_zero_ori)
        non_zero_inv = block_is_in(col_indices_ori, non_zero_ori)

        col_indices = torch.index_select(col_indices_ori, index=non_zero, dim=0)
        mask_values = torch.index_select(mask_values_ori, index=non_zero, dim=0)
        r_indices = torch.index_select(r_indices_ori, index=non_zero, dim=0)
        if weight is not None:
            C_ori = torch.index_select(weight, index=non_zero.to(weight.device), dim=0).to(A.device)
            C_ori = score_function_2(C_ori)

        col_indices_inv = torch.index_select(row_indices_ori, index=non_zero_inv, dim=0)
        mask_values_inv = torch.index_select(mask_values_ori, index=non_zero_inv, dim=0)
        r_indices_inv = torch.index_select(r_indices_ori, index=non_zero_inv, dim=0)
        if weight is not None:
            C_inv = torch.index_select(weight, index=non_zero_inv.to(weight.device), dim=0).to(A.device)
            C_inv = score_function_2(C_inv)

        w = torch.softmax(w / tau_1, dim=0)

        B_values = torch.index_select(w[:r_size, :], index=r_indices, dim=0)
        result_values = B_values.t() * mask_values.unsqueeze(0)
        if weight is not None: result_values = result_values * C_ori.t()
                                                                     
        col_indices_uni = torch.unique(col_indices)
        sorted_indices = torch.searchsorted(col_indices_uni, col_indices)

        index_ori = sorted_indices.long() * r_size + r_indices.long()
        type_ori = torch.unique(index_ori)
        type_index_ori = torch.searchsorted(type_ori, index_ori)
        result_values = create_compact(type_index_ori, result_values, type_ori.shape[0])
        col_indices = type_ori // r_size
        result = scatter_sum(result_values, col_indices, dim=1, dim_size=col_indices_uni.shape[0])
                                                 
                                                                        
        index = torch.ones_like(col_indices_uni) * i
        index = torch.stack([index, col_indices_uni], dim=0)
        indices_all.append(index)
        results_all.append(result)

                                                                    
        B_values_inv = torch.index_select(w[r_size: 2 * r_size, :], index=r_indices_inv, dim=0)
        result_values_inv = B_values_inv.t() * mask_values_inv.unsqueeze(0)
        if weight is not None: result_values_inv = result_values_inv * C_inv.t()
                                                                             
        col_indices_uni_inv = torch.unique(col_indices_inv)
        sorted_indices_inv = torch.searchsorted(col_indices_uni_inv, col_indices_inv)
        
        index_inv = sorted_indices_inv.long() * r_size + r_indices_inv.long()
        type_inv = torch.unique(index_inv)
        type_index_inv = torch.searchsorted(type_inv, index_inv)
        result_values_inv = create_compact(type_index_inv, result_values_inv, type_inv.shape[0])
        col_indices_inv = type_inv // r_size
        
        result_inv = scatter_sum(result_values_inv, col_indices_inv, dim=1, dim_size=col_indices_uni_inv.shape[0])
                                                     
                                                                                
        index = torch.ones_like(col_indices_uni_inv) * i
        index = torch.stack([index, col_indices_uni_inv], dim=0)
        indices_all_inv.append(index)
        results_all_inv.append(result_inv)

        if not wot_i:
            index = torch.ones_like(A[i]) * i
            index = torch.stack([index, A[i]], dim=0)
            indices_all_ind.append(index.unsqueeze(dim=-1))
            results_all_ind.append(w[-1:, :])
    i = torch.cat(indices_all, dim=-1)
    v = torch.cat(results_all, dim=-1).t()
    output_ori = torch.sparse_coo_tensor(i.long(), v, torch.Size([batch_size, E, L]))

    i = torch.cat(indices_all_inv, dim=-1)
    v = torch.cat(results_all_inv, dim=-1).t()
    output_inv = torch.sparse_coo_tensor(i.long(), v, torch.Size([batch_size, E, L]))

    if not wot_i:
        i = torch.cat(indices_all_ind, dim=-1)
        v = torch.cat(results_all_ind, dim=0)
        output_ind = torch.sparse_coo_tensor(i.long(), v, torch.Size([batch_size, E, L]))
    else:
        output_ind = None

    return output_ind, output_ori, output_inv

def sparse_matrix_multiply_L_sp_max(A, B, E, r_size, tau_1, is_training=False, dropout=None, is_max=False,
                                top_k=1000, topk_pruning=100000, wot_i=False, weight=None, use_topk=False):
    

    row_indices_ori, col_indices_ori, r_indices_ori, mask_values_ori, w_all = B
    indices_all, results_all = [], []
    indices_all_inv, results_all_inv = [], []
    indices_all_ind, results_all_ind = [], []
    A_batch = torch.unbind(A, dim=0)
    batch_size = len(A_batch)
    L = w_all.shape[1]
    for i in range(batch_size):
        w = w_all[i].t()
        A_ = A_batch[i].coalesce()
        A_indices = A_.indices()[0]
        A_values = A_.values()
        non_zero_ori = A_indices
        if use_topk:
            k_ = min(top_k, A_values.shape[0])
            topk = torch.topk(A_values.sum(1), k=k_)[1]
            if topk.shape[0] < non_zero_ori.shape[0]:
                non_zero_ori = A_indices[topk]
        non_zero = block_is_in(row_indices_ori, non_zero_ori)
        row_indices = torch.index_select(row_indices_ori, index=non_zero, dim=0)
        col_indices = torch.index_select(col_indices_ori, index=non_zero, dim=0)
        mask_values = torch.index_select(mask_values_ori, index=non_zero, dim=0)
        r_indices = torch.index_select(r_indices_ori, index=non_zero, dim=0)
        if weight is not None:
            C_ori = torch.index_select(weight, index=non_zero.to(weight.device), dim=0).to(A.device)
            C_ori = score_function_2(C_ori)

        non_zero_inv = block_is_in(col_indices_ori, non_zero_ori)
        row_indices_inv = torch.index_select(col_indices_ori, index=non_zero_inv, dim=0)
        col_indices_inv = torch.index_select(row_indices_ori, index=non_zero_inv, dim=0)
        mask_values_inv = torch.index_select(mask_values_ori, index=non_zero_inv, dim=0)
        r_indices_inv = torch.index_select(r_indices_ori, index=non_zero_inv, dim=0)
        if weight is not None:
            C_inv = torch.index_select(weight, index=non_zero_inv.to(weight.device), dim=0).to(A.device)
            C_inv = score_function_2(C_inv)

        w = torch.softmax(w / tau_1, dim=0)

        sorted_indices = torch.searchsorted(A_indices, row_indices)
        A_values_ori = torch.index_select(A_values, dim=0, index=sorted_indices)
        if use_topk:
            k_ = min(topk_pruning, A_values_ori.shape[0])
            A_values_ori_topk, A_values_ori_topk_indices = torch.topk(A_values_ori, k=k_, dim=0)
            B_values_ori = torch.index_select(w[:r_size, :], index=r_indices, dim=0)
            B_values_ori_topk = torch.gather(B_values_ori, index=A_values_ori_topk_indices, dim=0)
            mask_values_ori_topk = mask_values[A_values_ori_topk_indices]
            result_values_ori = A_values_ori_topk * B_values_ori_topk * mask_values_ori_topk
            if weight is not None:
                C_ori_topk = C_ori.squeeze(dim=1)[A_values_ori_topk_indices]
                result_values_ori = result_values_ori * C_ori_topk
                                                                            
            col_indices_ori_topk = col_indices[A_values_ori_topk_indices]
            col_indices_uni = torch.unique(col_indices_ori_topk)
            sorted_indices = torch.searchsorted(col_indices_uni, col_indices_ori_topk)

            index_ori = sorted_indices.long() * r_size + r_indices[A_values_ori_topk_indices].long()
            type_index_ori, type_ori = create_type_sp(index_ori)
            result_values_ori = create_compact(type_index_ori, result_values_ori.t(), type_ori.shape[-1])
            col_indices = type_ori // r_size
            if col_indices_uni.shape[0] == 0: continue
            result_ori = scatter_sum(result_values_ori, col_indices, dim=1, dim_size=col_indices_uni.shape[0]).t()
        else:
            B_values_ori = torch.index_select(w[:r_size, :], index=r_indices, dim=0)
            result_values_ori = A_values_ori * B_values_ori * mask_values.unsqueeze(dim=1)
            if weight is not None:
                result_values_ori = result_values_ori * C_ori.squeeze(dim=1)
                                                                            
            col_indices_uni = torch.unique(col_indices)
            sorted_indices = torch.searchsorted(col_indices_uni, col_indices)

            index_ori = sorted_indices.long() * r_size + r_indices.long()
            type_ori = torch.unique(index_ori)
            type_index_ori = torch.searchsorted(type_ori, index_ori)
            result_values_ori = create_compact(type_index_ori, result_values_ori.t(), type_ori.shape[0])
            col_indices = type_ori // r_size
            result_ori = scatter_sum(result_values_ori.t(), col_indices, dim=0, dim_size=col_indices_uni.shape[0])

                                                 
                                                                  
        index = torch.ones_like(col_indices_uni) * i
        index = torch.stack([index, col_indices_uni], dim=0)
        indices_all.append(index)
        results_all.append(result_ori)

        sorted_indices_inv = torch.searchsorted(A_indices, row_indices_inv)
        A_values_inv = torch.index_select(A_values, dim=0, index=sorted_indices_inv)
        if use_topk:
            k_ = min(topk_pruning, A_values_inv.shape[0])
            A_values_inv_topk, A_values_inv_topk_indices = torch.topk(A_values_inv, k=k_, dim=0)
            B_values_inv = torch.index_select(w[r_size: 2 * r_size, :], index=r_indices_inv, dim=0)
            B_values_inv_topk = torch.gather(B_values_inv, index=A_values_inv_topk_indices, dim=0)
            mask_values_inv_topk = mask_values_inv[A_values_inv_topk_indices]
            result_values_inv = A_values_inv_topk * B_values_inv_topk * mask_values_inv_topk
            if weight is not None:
                C_inv_topk = C_inv.squeeze(dim=1)[A_values_inv_topk_indices]
                result_values_inv = result_values_inv * C_inv_topk
                                                                            
            col_indices_inv_topk = col_indices_inv[A_values_inv_topk_indices]
            col_indices_uni_inv = torch.unique(col_indices_inv_topk)
            sorted_indices_inv = torch.searchsorted(col_indices_uni_inv, col_indices_inv_topk)

            index_inv = sorted_indices_inv.long() * r_size + r_indices_inv[A_values_inv_topk_indices].long()
            type_index_inv, type_inv = create_type_sp(index_inv)
            result_values_inv = create_compact(type_index_inv, result_values_inv.t(), type_inv.shape[-1])
            col_indices_inv = type_inv // r_size
            if col_indices_uni_inv.shape[0] == 0: continue
            result_inv = scatter_sum(result_values_inv, col_indices_inv, dim=1,
                                     dim_size=col_indices_uni_inv.shape[0]).t()
        else:
            B_values_inv = torch.index_select(w[r_size: 2 * r_size, :], index=r_indices_inv, dim=0)
            result_values_inv = A_values_inv * B_values_inv * mask_values_inv.unsqueeze(dim=1)
            if weight is not None:
                result_values_inv = result_values_inv * C_inv.squeeze(dim=1)
                                                                            
            col_indices_uni_inv = torch.unique(col_indices_inv)
            sorted_indices_inv = torch.searchsorted(col_indices_uni_inv, col_indices_inv)

            index_inv = sorted_indices_inv.long() * r_size + r_indices_inv.long()
            type_inv = torch.unique(index_inv)
            type_index_inv = torch.searchsorted(type_inv, index_inv)
            result_values_inv = create_compact(type_index_inv, result_values_inv.t(), type_inv.shape[0])
            col_indices_inv = type_inv // r_size
            result_inv = scatter_sum(result_values_inv.t(), col_indices_inv, dim=0,
                                 dim_size=col_indices_uni_inv.shape[0])

        index = torch.ones_like(col_indices_uni_inv) * i
        index = torch.stack([index, col_indices_uni_inv], dim=0)
        indices_all_inv.append(index)
        results_all_inv.append(result_inv)

        if not wot_i:
            result_ind = A_values * w[-1, :].unsqueeze(dim=0)
            index = torch.ones_like(A_indices) * i
            index = torch.stack([index, A_indices], dim=0)
            indices_all_ind.append(index)
            results_all_ind.append(result_ind)
    if len(indices_all) == 0:
        i = torch.LongTensor([0]).to(A.device).unsqueeze(0).repeat(2, 1)
        v = i.float().unsqueeze(-1).repeat(1, L)
        output_ori = torch.sparse_coo_tensor(i.long(), v, torch.Size([batch_size, E, L]))
    else:
        i = torch.cat(indices_all, dim=-1)
        v = torch.cat(results_all, dim=0)
        output_ori = torch.sparse_coo_tensor(i.long(), v, torch.Size([batch_size, E, L]))

    if len(indices_all_inv) == 0:
        i = torch.LongTensor([0]).to(A.device).unsqueeze(0).repeat(2, 1)
        v = i.float().unsqueeze(-1).repeat(1, L)
        output_inv = torch.sparse_coo_tensor(i.long(), v, torch.Size([batch_size, E, L]))
    else:
        i = torch.cat(indices_all_inv, dim=-1)
        v = torch.cat(results_all_inv, dim=0)
        output_inv = torch.sparse_coo_tensor(i.long(), v, torch.Size([batch_size, E, L]))

    if not wot_i:
        i = torch.cat(indices_all_ind, dim=-1)
        v = torch.cat(results_all_ind, dim=0)
        output_ind = torch.sparse_coo_tensor(i.long(), v, torch.Size([batch_size, E, L]))
    else:
        output_ind = None

    return output_ind, output_ori, output_inv


def match_constants( s, t, e2triple, constant, constant_inv, gate, gate_inv, flag):
    constant = torch.sigmoid(constant[t].t().unsqueeze(dim=0))
    gate = torch.sigmoid(gate[t]).unsqueeze(dim=0).unsqueeze(dim=-1)
    if flag:
        constant = torch.sigmoid(constant_inv[t].t().unsqueeze(dim=0))
        gate = torch.sigmoid(gate_inv[t]).unsqueeze(dim=0).unsqueeze(dim=-1)
    s_ori = s
    s *= gate
    s[:, :, e2triple[-1]] += constant * s_ori[:, :, e2triple[-1]] * (1 - gate)
    return s


def match_neibour( A, B, h, n, batch_size):
    row_indices, col_indices, r_indices = B
    results = []
    for i in range(batch_size):
        non_zero = A[i]
        indices_row = block_is_in(row_indices, non_zero)
        r_indices_row = torch.index_select(r_indices, index=indices_row, dim=0)
        indices_row_uni = torch.unique(r_indices_row, dim=0)
                                                                               
        B_values = torch.range(0, n - 1).to(A.device)
        B_values = torch.isin(B_values, indices_row_uni).float()

        indices_col = block_is_in(col_indices, non_zero)
        r_indices_col = torch.index_select(r_indices, index=indices_col, dim=0)
        indices_col_uni = torch.unique(r_indices_col, dim=0)
        C_values = torch.range(0, n - 1).to(A.device)
        C_values = torch.isin(C_values, indices_col_uni).float()
        D_values = torch.ones(1).to(A.device)
                                                                                  
                                                                                                                     
        constraints = torch.cat([B_values, C_values, D_values], dim=0)
        constraints = torch.einsum('n,ln->l', constraints, h[i])
                                                                                     
        results.append(constraints)
    return torch.cat(results, dim=0)

def norm_sp(s):
    batches = torch.unbind(s)
    batches_new = []
    for item in batches:
        s_i = item.coalesce()
        values = s_i.values()
        values = values / values.sum(dim=0, keepdims=True).clamp(1e-7)
        sp_new = torch.sparse_coo_tensor(s_i.indices().long(), values, s_i.shape)
        batches_new.append(sp_new)
    return torch.stack(batches_new, dim=0)

def sum_sp(s):
    A = s.coalesce()
    shape = A.shape
    indices = A.indices()
    values = A.values().sum(dim=-1)
    return torch.sparse_coo_tensor(indices.long(), values, torch.Size([shape[0], shape[1]]))

def max_sp(s):
    A = s.coalesce()
    shape = A.shape
    indices = A.indices()
    values = A.values().max(dim=-1)[0]
    return torch.sparse_coo_tensor(indices.long(), values, torch.Size([shape[0], shape[1]]))

def score_function(x):
                                                                 
                                         
    return torch.sigmoid(x)
                                                                              
                                            
                                        

def score_function_2(x):
                                                                 
                                         
    return x

def mask_data(values, indices, score):
                           
                               
    values[indices] = score
        
def sym_update(set1, set2, set3):
    S_sym = set()
    for item in set1:
        S_sym.add((item[2], item[1], item[0]))
    facts_to_process = set2.intersection(S_sym)
    set1 = set1 - facts_to_process
    
    set3.update(facts_to_process)
    
    return set1, set2, set3

def calculate_hits(state, top_k, batches, graph, graph_inv, dataset, hits):
    scores, indices = torch.topk(state, k=top_k)
    scores = scores.detach().cpu()
    indices = indices.detach().cpu()
    for j, items in enumerate(batches):
        h, r, t = items['h'], items['r'], items['t']
        indices_list = list(indices[j])
        if t in indices_list:
            truth_index = indices_list.index(t)
            if r < dataset.relation_size:
                scores_tail = scores[j, :].clone()
                if (h, r) in graph.indices.keys():
                    group = graph.get_group((h, r))
                    truths = set(group[2].values.tolist())
                    for tail in indices_list:
                        if tail.item() not in truths: continue
                        if tail.item() == t: continue
                        scores_tail[indices_list.index(tail)] = -1e20
                    sorted = torch.topk(scores_tail, k=10)
                    indices_ = list(sorted[1])
                    if truth_index in indices_:
                        rank = indices_.index(truth_index)
                        hits[rank:] += 1
            else:
                scores_head = scores[j, :].clone()
                if (h, r - dataset.relation_size) in graph_inv.indices.keys():
                    group = graph_inv.get_group((h, r - dataset.relation_size))
                    truths = set(group[0].values.tolist())
                    for head in indices_list:
                        if head.item() not in truths: continue
                        if head.item() == t: continue
                        scores_head[indices_list.index(head)] = -1e20
                sorted = torch.topk(scores_head, k=10)
                indices_ = list(sorted[1])
                if truth_index in indices_:
                    rank = indices_.index(truth_index)
                    hits[rank:] += 1

def calculate_hits_full(state, batches, e2triple, triple2e, r2triple, graph, graph_inv, dataset, hits, use_eql=False):
    scores_all = state
    mrr = 0
    input_x = []
    input_y = []
    r_mask_x = []
    r_mask_y = []
    truths = []
    truths_eval = []
    for j, items in enumerate(batches):
        h, r, t = items['h'], items['r'], items['t']
        scores = scores_all[j]
        truth_score = scores[t].clone()
        truths.append(truth_score)
        if r < dataset.relation_size:
            input_x.append([j, h])
            mask = (r2triple[0] == r).float()
            r_mask_x.append(mask)
            if (h, r) in graph.indices.keys():
                group = graph.get_group((h, r))
                eval = torch.from_numpy(group[2].values).to(scores_all.device).long()
            else:
                eval = torch.LongTensor([]).to(scores_all.device)
        else:
            input_y.append([j, h])
            r_ = r - dataset.relation_size
            mask = (r2triple[0] == r_).float()
            r_mask_y.append(mask)
            if (h, r - dataset.relation_size) in graph_inv.indices.keys():
                group = graph_inv.get_group((h, r - dataset.relation_size))
                eval = torch.from_numpy(group[0].values).to(scores_all.device).long()
            else:
                eval = torch.LongTensor([]).to(scores_all.device)

        eval = torch.sparse_coo_tensor(
            torch.stack([eval, torch.zeros_like(eval)], dim=0).long(),
            torch.ones_like(eval),
            torch.Size([scores_all.shape[1], 1]),
        )
        truths_eval.append(eval)

    truths_eval = torch.stack(truths_eval, dim=0).to_dense().squeeze(dim=-1)
    if len(r_mask_x) != 0:
        r_mask_x = torch.stack(r_mask_x, dim=0)
        input_x = torch.LongTensor(input_x).to(scores_all.device)
        input_x_oh = torch.nn.functional.one_hot(input_x[:, 1], scores_all.shape[1])
        x_facts = torch.index_select(input_x_oh, index=e2triple[0], dim=1) * r_mask_x
        x_mask = scatter_sum(x_facts, triple2e[1].long(), dim=1, dim_size=scores_all.shape[-1])
        scores_all[input_x[:, 0]] = scores_all[input_x[:, 0]] - x_mask * 1e20 - truths_eval[input_x[:, 0]] * 1e20
    if len(r_mask_y) != 0:
        r_mask_y = torch.stack(r_mask_y, dim=0)
        input_y = torch.LongTensor(input_y).to(scores_all.device)
        input_y_oh = torch.nn.functional.one_hot(input_y[:, 1], scores_all.shape[1])
        y_facts = torch.index_select(input_y_oh, index=triple2e[1], dim=1) * r_mask_y
        y_mask = scatter_sum(y_facts, e2triple[0].long(), dim=1, dim_size=scores_all.shape[-1])
        scores_all[input_y[:, 0]] = scores_all[input_y[:, 0]] - y_mask * 1e20 - truths_eval[input_y[:, 0]] * 1e20

    for j, items in enumerate(batches):
        truth_score = truths[j]
        scores = scores_all[j]
        m = (scores > truth_score).int().sum()
        if use_eql:
            n = (scores == truth_score).int().sum() + 1
            rank = m + (n + 1) / 2
                                           
        else:
            rank = m + 1
        hits[round(rank.item()) - 1:] += 1
        mrr += 1 / rank.item()
    return mrr
