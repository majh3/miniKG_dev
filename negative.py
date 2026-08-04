

from __future__ import annotations

import torch

try:
    from .facts import encode_membership_keys, fact_membership_from_sorted_keys
except ImportError:                           
    from facts import encode_membership_keys, fact_membership_from_sorted_keys


def parse_relation_subset_tensor(value, relation_count: int, device: torch.device):
    if value is None or str(value).strip() == "":
        return None
    ids = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        rel_id = int(part)
        if 0 <= rel_id < relation_count:
            ids.append(rel_id)
    if not ids:
        return None
    return torch.tensor(sorted(set(ids)), dtype=torch.long, device=device)


def relation_subset_tensor_for_targets(value, relation_count: int, device: torch.device, targets: torch.Tensor):
    if value is not None and str(value).strip() == "batch":
        if targets is None or targets.numel() == 0:
            return None
        rel_ids = torch.unique(targets[:, 1].long().to(device=device))
        return rel_ids[(rel_ids >= 0) & (rel_ids < relation_count)]
    return parse_relation_subset_tensor(value, relation_count, device)


def sample_corrupt_targets(targets: torch.Tensor, truth_by_query: dict, entity_count: int, generator: torch.Generator):
    rows = []
    for h, r, t in targets.detach().cpu().tolist():
        true_tails = truth_by_query.get((int(h), int(r)), set())
        for _ in range(64):
            false_t = int(torch.randint(0, entity_count, (1,), generator=generator, device=targets.device).item())
            if false_t not in true_tails:
                rows.append((int(h), int(r), false_t))
                break
        else:
            false_t = (int(t) + 1) % entity_count
            while false_t in true_tails:
                false_t = (false_t + 1) % entity_count
            rows.append((int(h), int(r), false_t))
    return torch.tensor(rows, dtype=torch.long, device=targets.device)


def rule_generated_negative_targets(
    model,
    facts_tensor: torch.Tensor,
    indices: torch.Tensor,
    entity_count: int,
    relation_count: int,
    sorted_true_keys: torch.Tensor,
    key_mode: str,
    relation_ids=None,
    topk: int = 1,
):
    source = facts_tensor[indices.long()]
    batch_size = source.shape[0]
    device = facts_tensor.device
    rel_ids = relation_ids
    if rel_ids is None:
        rel_ids = torch.arange(relation_count, device=device)
    else:
        rel_ids = rel_ids.to(device=device, dtype=torch.long)
    candidate_count = int(rel_ids.numel())
    h = source[:, 0:1]
    source_r = source[:, 1].long()
    t = source[:, 2:3]

    forward = torch.stack(
        [h.expand(batch_size, candidate_count), rel_ids.expand(batch_size, candidate_count), t.expand(batch_size, candidate_count)],
        dim=2,
    )
    backward = torch.stack(
        [t.expand(batch_size, candidate_count), rel_ids.expand(batch_size, candidate_count), h.expand(batch_size, candidate_count)],
        dim=2,
    )
    candidates = torch.cat([forward, backward], dim=1)
    candidate_keys = encode_membership_keys(candidates.reshape(-1, 3), entity_count, relation_count, key_mode).view(batch_size, -1)
    if sorted_true_keys.device != candidate_keys.device:
        is_true = fact_membership_from_sorted_keys(candidate_keys.cpu(), sorted_true_keys).to(device=device)
    else:
        is_true = fact_membership_from_sorted_keys(candidate_keys, sorted_true_keys)

    rule_prob = torch.softmax(model.build_rule_logits(rel_ids) / model.tau_1, dim=-1)
    rule_score = rule_prob[:, 0].amax(dim=1).detach()
    forward_scores = rule_score[:, source_r].transpose(0, 1)
    backward_scores = rule_score[:, source_r + relation_count].transpose(0, 1)
    scores = torch.cat([forward_scores, backward_scores], dim=1).masked_fill(is_true, -1.0)
    k = max(1, min(int(topk), scores.shape[1]))
    if k == 1:
        best = scores.argmax(dim=1)
        negatives = candidates[torch.arange(batch_size, device=device), best]
        fallback_mask = scores.max(dim=1).values < 0
    else:
        top_values, top_indices = torch.topk(scores, k=k, dim=1)
        valid = top_values >= 0
        negatives = candidates[
            torch.arange(batch_size, device=device).view(-1, 1).expand_as(top_indices),
            top_indices,
        ][valid]
        fallback_mask = ~valid.any(dim=1)
    if bool(fallback_mask.any().item()):
        fallback = source[fallback_mask].clone().to(dtype=negatives.dtype)
        fallback[:, 2] = (fallback[:, 2] + 1) % entity_count
        if k == 1:
            negatives[fallback_mask] = fallback
        else:
            negatives = torch.cat([negatives, fallback], dim=0)
    return negatives
