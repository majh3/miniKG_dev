"""Fact tensor utilities and membership keys."""

from __future__ import annotations

import numpy as np
import torch


def fact_count(facts) -> int:
    return int(facts.shape[0]) if isinstance(facts, np.ndarray) else len(facts)


def facts_to_tensor(facts, device: torch.device) -> torch.Tensor:
    if isinstance(facts, np.ndarray):
        base = torch.from_numpy(np.asarray(facts))
        return base.to(device=device, dtype=torch.int32)
    return torch.tensor(facts, dtype=torch.long, device=device)


def infer_entity_relation_count(facts) -> tuple[int, int]:
    if isinstance(facts, np.ndarray):
        entity_count = int(max(facts[:, 0].max(), facts[:, 2].max())) + 1
        relation_count = int(facts[:, 1].max()) + 1
        return entity_count, relation_count
    entity_count = max(max(h, t) for h, _r, t in facts) + 1
    relation_count = max(r for _h, r, _t in facts) + 1
    return entity_count, relation_count


def encode_fact_keys(facts_tensor: torch.Tensor, entity_count: int, relation_count: int) -> torch.Tensor:
    return (facts_tensor[:, 0] * relation_count + facts_tensor[:, 1]) * entity_count + facts_tensor[:, 2]


def exact_fact_key_is_safe(entity_count: int, relation_count: int) -> bool:
    return (entity_count - 1) * relation_count * entity_count + (relation_count - 1) * entity_count + (entity_count - 1) <= (2**63 - 1)


def encode_fact_hash_keys(facts_tensor: torch.Tensor) -> torch.Tensor:
    facts_long = facts_tensor.long()
    return (
        facts_long[:, 0] * 6364136223846793005
        ^ facts_long[:, 1] * 1442695040888963407
        ^ facts_long[:, 2] * 22695477
    )


def encode_membership_keys(facts_tensor: torch.Tensor, entity_count: int, relation_count: int, mode: str) -> torch.Tensor:
    if mode == "auto":
        mode = "exact" if exact_fact_key_is_safe(entity_count, relation_count) else "hash"
    if mode == "exact":
        if not exact_fact_key_is_safe(entity_count, relation_count):
            raise ValueError("exact fact membership key overflows int64; use hash mode")
        return encode_fact_keys(facts_tensor.long(), entity_count, relation_count)
    if mode == "hash":
        return encode_fact_hash_keys(facts_tensor)
    raise ValueError(f"unknown fact membership key mode: {mode}")


def fact_membership_from_sorted_keys(candidate_keys: torch.Tensor, sorted_true_keys: torch.Tensor) -> torch.Tensor:
    if sorted_true_keys.numel() == 0:
        return torch.zeros_like(candidate_keys, dtype=torch.bool)
    positions = torch.searchsorted(sorted_true_keys, candidate_keys)
    in_bounds = positions < sorted_true_keys.numel()
    safe_positions = positions.clamp(max=max(sorted_true_keys.numel() - 1, 0))
    return in_bounds & (sorted_true_keys[safe_positions] == candidate_keys)
