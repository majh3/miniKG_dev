"""Streaming candidate generation for the single champion decode path."""

from __future__ import annotations

from dataclasses import dataclass
import json
from types import SimpleNamespace
from typing import Iterator, Sequence

import numpy as np
import torch

try:
    from .drum import SupplyGatedDRUM
    from .graph import Graph
    from .proof import (
        bound_sparse_scores,
        compact_kernel_graph_for_decode,
        kernel_sparse_scores,
        sparse_candidate_rows,
    )
except ImportError:  # direct script execution
    from drum import SupplyGatedDRUM
    from graph import Graph
    from proof import (
        bound_sparse_scores,
        compact_kernel_graph_for_decode,
        kernel_sparse_scores,
        sparse_candidate_rows,
    )


@dataclass(frozen=True)
class DirectionCandidateRow:
    """One ordered query row; arrays are CPU NumPy arrays."""

    relation: int
    query_head: int
    ordered_tails: np.ndarray
    ordered_scores: np.ndarray
    true_tails: np.ndarray
    base_tails: np.ndarray

def oriented_relation_arrays(
    facts: np.ndarray,
    relation: int,
    orientation_inverse: np.ndarray | None,
    precomputed_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if precomputed_idx is None:
        idx = np.flatnonzero(np.asarray(facts[:, 1], dtype=np.int64) == int(relation))
    else:
        idx = np.asarray(precomputed_idx, dtype=np.int64)
    raw_heads = np.asarray(facts[:, 0][idx], dtype=np.int64)
    raw_tails = np.asarray(facts[:, 2][idx], dtype=np.int64)
    if orientation_inverse is not None and bool(orientation_inverse[int(relation)]):
        return idx, raw_tails, raw_heads
    return idx, raw_heads, raw_tails


def build_relation_row_index(
    facts: np.ndarray,
    relation_ids: list[int] | tuple[int, ...],
    *,
    chunk_size: int = 5_000_000,
) -> dict[int, np.ndarray]:
    """One chunked pass over facts to collect row indices for selected relations.

    Replaces per-relation full-table ``flatnonzero`` scans (56 × 305M on FB head70),
    which silently wedged the first decode relation for 10+ minutes.
    """
    rel_ids = [int(r) for r in relation_ids]
    wanted = set(rel_ids)
    if not wanted:
        return {}
    n = int(facts.shape[0])
    chunk_size = max(1, int(chunk_size))
    buckets: dict[int, list[np.ndarray]] = {r: [] for r in wanted}
    rel_col = np.asarray(facts[:, 1], dtype=np.int64)
    import json as _json

    print(
        _json.dumps(
            {
                "event": "relation_index_build_start",
                "facts": n,
                "relations": len(wanted),
                "chunk_size": chunk_size,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sl = rel_col[start:end]
        for r in wanted:
            local = np.flatnonzero(sl == r)
            if local.size:
                buckets[r].append(local.astype(np.int64, copy=False) + start)
        if start == 0 or end == n or (start // chunk_size) % 10 == 0:
            print(
                _json.dumps(
                    {
                        "event": "relation_index_build_progress",
                        "done": end,
                        "total": n,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    out: dict[int, np.ndarray] = {}
    for r, parts in buckets.items():
        if parts:
            out[r] = np.concatenate(parts, axis=0)
        else:
            out[r] = np.empty((0,), dtype=np.int64)
    print(
        _json.dumps(
            {
                "event": "relation_index_build_done",
                "relations": len(out),
                "rows": int(sum(int(v.size) for v in out.values())),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return out


def hard_supply_entity_set(
    facts: np.ndarray,
    hard_supply_np: np.ndarray,
) -> set[int]:
    rows = facts[hard_supply_np]
    if rows.size == 0:
        return set()
    return set(int(x) for x in np.concatenate([rows[:, 0], rows[:, 2]]).tolist())


def hard_supply_head_set(
    facts: np.ndarray,
    hard_supply_np: np.ndarray,
) -> set[int]:
    rows = facts[hard_supply_np]
    if rows.size == 0:
        return set()
    return set(int(x) for x in rows[:, 0].tolist())




def materialize_supply_proof_graph(
    facts: np.ndarray,
    hard_supply: torch.Tensor,
    device: torch.device,
) -> tuple[Graph, torch.Tensor]:
    """Build the physical proof graph represented by a hard supply decision.

    FastLog applies its per-node edge cap before multiplying edge scores.  A
    full graph with zero-weight dropped edges is therefore not equivalent to a
    graph containing only stored supply: dead edges can consume the cap.  The
    deployment graph must contain exactly the stored premises, with unit edge
    weights, which is also the historical YAGO replay contract.
    """
    supply = hard_supply.detach().cpu().numpy().astype(np.bool_, copy=False)
    rows = np.asarray(facts[supply], dtype=np.int64)
    graph = Graph(torch.as_tensor(rows, dtype=torch.long, device=device))
    graph_supply = torch.ones(rows.shape[0], dtype=torch.bool, device=device)
    return graph, graph_supply






def relation_grouped_arrays(
    facts: np.ndarray,
    relation: int,
    hard_supply_np: np.ndarray,
    orientation_inverse: np.ndarray | None,
    precomputed_idx: np.ndarray | None = None,
) -> tuple[int, int, dict[int, np.ndarray], dict[int, np.ndarray], set[int]]:
    """Return deduplicated truth/supply tails grouped by oriented query head."""
    rel_idx, heads, tails = oriented_relation_arrays(
        facts,
        relation,
        orientation_inverse,
        precomputed_idx=precomputed_idx,
    )
    supply = np.asarray(hard_supply_np[rel_idx], dtype=np.bool_)
    return group_relation_pairs(heads, tails, supply)


def group_relation_pairs(
    heads: np.ndarray,
    tails: np.ndarray,
    supply: np.ndarray,
) -> tuple[int, int, dict[int, np.ndarray], dict[int, np.ndarray], set[int]]:
    """Group one already-sliced relation without scanning the full fact table."""
    heads = np.asarray(heads, dtype=np.int64).ravel()
    tails = np.asarray(tails, dtype=np.int64).ravel()
    supply = np.asarray(supply, dtype=np.bool_).ravel()
    if heads.shape != tails.shape or heads.shape != supply.shape:
        raise ValueError("relation heads/tails/supply arrays must have equal shape")
    if heads.size == 0:
        return 0, 0, {}, {}, set()

    # lexsort + consecutive unique is much faster than np.unique(pairs, axis=0)
    # on multi-10M-row FB relations (the previous first-relation wedge).
    order = np.lexsort((tails, heads))
    h = heads[order]
    t = tails[order]
    s = supply[order]
    n = int(h.size)
    first = np.empty(n, dtype=np.bool_)
    first[0] = True
    if n > 1:
        first[1:] = (h[1:] != h[:-1]) | (t[1:] != t[:-1])
    uniq_idx = np.flatnonzero(first)
    unique_heads = h[uniq_idx]
    unique_tails = t[uniq_idx]
    # OR-reduce supply flags onto unique pairs (groups are contiguous after sort).
    supply_any = np.zeros(uniq_idx.size, dtype=np.bool_)
    # segment ids 0..U-1 for each row
    seg = np.cumsum(first) - 1
    np.logical_or.at(supply_any, seg, s)

    def grouped_from_sorted(pair_heads: np.ndarray, pair_tails: np.ndarray) -> dict[int, np.ndarray]:
        result: dict[int, np.ndarray] = {}
        if pair_heads.size == 0:
            return result
        # pair_heads already sorted
        uniq_h, starts = np.unique(pair_heads, return_index=True)
        bounds = np.append(starts, pair_heads.shape[0])
        for index, head in enumerate(uniq_h):
            result[int(head)] = np.ascontiguousarray(
                pair_tails[bounds[index] : bounds[index + 1]],
                dtype=np.int64,
            )
        return result

    truth_by_head = grouped_from_sorted(unique_heads, unique_tails)
    base_mask = supply_any
    base_heads_arr = unique_heads[base_mask]
    base_tails_arr = unique_tails[base_mask]
    base_by_head = grouped_from_sorted(base_heads_arr, base_tails_arr)
    base_heads = set(int(v) for v in np.unique(base_heads_arr)) if base_heads_arr.size else set()
    return (
        int(unique_heads.size),
        int(supply_any.sum()),
        truth_by_head,
        base_by_head,
        base_heads,
    )


def query_heads_for_relation(
    source: str,
    facts: np.ndarray,
    relation: int,
    hard_supply_np: np.ndarray,
    orientation_inverse: np.ndarray | None,
    base_heads: set[int],
) -> list[int]:
    if source == "target_truth_heads":
        _indices, heads, _tails = oriented_relation_arrays(
            facts,
            relation,
            orientation_inverse,
        )
        return sorted(set(int(head) for head in heads.tolist()))
    if source == "target_supply_heads":
        return sorted(base_heads)
    if source == "supply_heads":
        return sorted(hard_supply_head_set(facts, hard_supply_np))
    if source == "supply_entities":
        return sorted(hard_supply_entity_set(facts, hard_supply_np))
    raise ValueError(
        "threshold decode supports supply_entities/supply_heads/"
        f"target_supply_heads/target_truth_heads, got {source!r}"
    )


def ordered_tail_scores(
    candidate_tails,
    candidate_scores,
    candidate_cap: int,
) -> tuple[np.ndarray, np.ndarray]:
    tails = np.asarray(candidate_tails, dtype=np.int64)
    scores = np.asarray(candidate_scores, dtype=np.float32)
    if tails.size == 0:
        return tails[:0], scores[:0]
    order = np.lexsort((tails, -scores))
    if candidate_cap > 0:
        order = order[: int(candidate_cap)]
    return tails[order], scores[order]


def maybe_compact_graph_for_final_decode(
    model: SupplyGatedDRUM,
    graph: Graph,
    args: SimpleNamespace,
) -> None:
    """Drop immutable raw topology once, only after training has ended."""
    if not bool(getattr(args, "_decode_graph_compaction_allowed", False)):
        return
    if bool(getattr(args, "_decode_graph_compaction_attempted", False)):
        return
    args._decode_graph_compaction_attempted = True
    if getattr(model, "edge_weight_mask", None) is not None:
        args._decode_graph_compaction_skipped = "dynamic_edge_mask"
        return
    released = compact_kernel_graph_for_decode(model, graph)
    args._decode_graph_compaction_released_bytes = int(released)
    print(
        json.dumps(
            {
                "event": "decode_graph_compacted",
                "released_storage_bytes": int(released),
                "scope": "final_decode_only",
            },
            sort_keys=True,
        ),
        flush=True,
    )


def iter_decode_candidate_batches(
    model: SupplyGatedDRUM,
    heads: torch.Tensor,
    relations: torch.Tensor,
    graph: Graph,
    args: SimpleNamespace,
):
    """Yield CPU candidate rows without retaining a full outer proof batch."""
    total = int(heads.shape[0])
    for row_start in range(0, total, max(total, 1)):
        row_end = min(row_start + max(total, 1), total)
        raw = kernel_sparse_scores(
            model,
            heads[row_start:row_end],
            relations[row_start:row_end],
            graph,
            args,
            is_training=False,
        )
        maybe_compact_graph_for_final_decode(model, graph, args)
        proof = bound_sparse_scores(raw)
        candidate_rows = sparse_candidate_rows(proof)
        del raw, proof
        yield row_start, row_end, candidate_rows




def iter_direction_candidate_rows(
    model: SupplyGatedDRUM,
    relation: int,
    query_heads: Sequence[int],
    truth_by_head: dict[int, np.ndarray],
    base_by_head: dict[int, np.ndarray],
    graph: Graph,
    args: SimpleNamespace,
) -> Iterator[DirectionCandidateRow]:
    """Stream ordered candidate rows for one relation."""
    heads = list(query_heads)
    batch_size = 4096 if args.dataset == "family" else 1024
    if batch_size <= 0:
        raise ValueError("decode_batch_size must be positive")
    device = model.weight_param.device
    empty_ids = np.empty(0, dtype=np.int64)

    for start in range(0, len(heads), batch_size):
        batch_heads = heads[start : start + batch_size]
        heads_tensor = torch.tensor(
            batch_heads,
            dtype=torch.long,
            device=device,
        )
        relations_tensor = torch.full(
            (len(batch_heads),),
            int(relation),
            dtype=torch.long,
            device=device,
        )
        for row_start, row_end, candidate_rows in iter_decode_candidate_batches(
            model,
            heads_tensor,
            relations_tensor,
            graph,
            args,
        ):
            for row_index, query_head in enumerate(batch_heads[row_start:row_end]):
                tails, scores = candidate_rows.get(row_index, ([], []))
                true_tails = truth_by_head.get(int(query_head), empty_ids)
                ordered_tails, ordered_scores = ordered_tail_scores(tails, scores, 0)
                yield DirectionCandidateRow(
                    relation=int(relation),
                    query_head=int(query_head),
                    ordered_tails=ordered_tails,
                    ordered_scores=ordered_scores,
                    true_tails=true_tails,
                    base_tails=base_by_head.get(int(query_head), empty_ids),
                )
