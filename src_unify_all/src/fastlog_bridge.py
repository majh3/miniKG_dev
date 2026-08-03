"""Minimal FastLog bridge used by the standalone MiniKG runtime."""

from __future__ import annotations

import torch

from .fastlog_kernel.fastlog_kernel import (
    FastLogFunctionSparse3DTopK,
    build_csr_structure,
)


def _sparse_coo_unique(idx, val, shape):
    return torch.sparse_coo_tensor(idx, val, shape, is_coalesced=True)


def sum_sp_3d(s):
    s = s.coalesce()
    idx = s.indices()
    val = s.values()
    batch_size, _, entity_count = s.shape
    idx2 = torch.stack([idx[0], idx[2]], dim=0)
    return torch.sparse_coo_tensor(idx2, val, [batch_size, entity_count]).coalesce()


def sum_sparse3d_concat(*states):
    states = [state.coalesce() for state in states]
    non_empty = [state for state in states if state._nnz() > 0]
    if not non_empty:
        empty_i = torch.empty(0, device=states[0].device, dtype=torch.long)
        empty_v = torch.empty(
            0,
            device=states[0].device,
            dtype=states[0].values().dtype,
        )
        empty_idx = torch.stack([empty_i, empty_i, empty_i], dim=0)
        return _sparse_coo_unique(empty_idx, empty_v, states[0].shape)
    if len(non_empty) == 1:
        return non_empty[0]
    idx = torch.cat([state.indices() for state in non_empty], dim=1)
    val = torch.cat([state.values() for state in non_empty], dim=0)
    return torch.sparse_coo_tensor(idx, val, non_empty[0].shape).coalesce()


__all__ = [
    "FastLogFunctionSparse3DTopK",
    "build_csr_structure",
    "sum_sp_3d",
    "sum_sparse3d_concat",
]
