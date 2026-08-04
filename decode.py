

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

try:
    from .decode_core import (
        build_relation_row_index,
        iter_direction_candidate_rows,
        materialize_supply_proof_graph,
        prepare_rule_first_hop_prune,
        query_heads_for_relation,
        relation_grouped_arrays,
        rule_first_hop_query_heads,
    )
    from .drum import SupplyGatedDRUM
    from .facts import fact_count
    from .graph import Graph
    from .metrics import DecodeMetrics
    from .proof import hard_supply_context
except ImportError:                           
    from decode_core import (
        build_relation_row_index,
        iter_direction_candidate_rows,
        materialize_supply_proof_graph,
        prepare_rule_first_hop_prune,
        query_heads_for_relation,
        relation_grouped_arrays,
        rule_first_hop_query_heads,
    )
    from drum import SupplyGatedDRUM
    from facts import fact_count
    from graph import Graph
    from metrics import DecodeMetrics
    from proof import hard_supply_context


def _oracle_best_k(
    not_base: np.ndarray,
    in_true: np.ndarray,
    base_true: int,
) -> tuple[int, int]:
    pass                                                                        
    n = int(not_base.size)
    new_true_pos = not_base & in_true
    prefix_new = np.empty(n + 1, dtype=np.int64)
    prefix_new_true = np.empty(n + 1, dtype=np.int64)
    prefix_new[0] = 0
    prefix_new_true[0] = 0
    np.cumsum(not_base, dtype=np.int64, out=prefix_new[1:])
    np.cumsum(new_true_pos, dtype=np.int64, out=prefix_new_true[1:])

    true_count = int(base_true) + prefix_new_true
    wrong = prefix_new - prefix_new_true
    gain = true_count - wrong
    gain_mask = gain == gain.max()
    true_mask = gain_mask & (
        true_count
        == np.where(gain_mask, true_count, np.iinfo(np.int64).min).max()
    )
    wrong_mask = true_mask & (
        wrong == np.where(true_mask, wrong, np.iinfo(np.int64).max).min()
    )
    best_k = int(np.argmax(wrong_mask))
    return best_k, int(wrong[best_k])


class _OracleRelationAccumulator:
    pass                                                               

    def __init__(self) -> None:
        self.candidate_total = 0
        self.max_candidate_count = 0
        self.query_count = 0
        self.selected = 0
        self.selected_wrong = 0
        self.new = 0
        self.new_true = 0

    def add_row(
        self,
        ordered_tails: np.ndarray,
        true_tails: np.ndarray,
        base_tails: np.ndarray,
    ) -> int:
        n = int(ordered_tails.size)
        self.candidate_total += n
        self.max_candidate_count = max(self.max_candidate_count, n)

        base_tails = np.asarray(base_tails, dtype=np.int64)
        true_tails = np.asarray(true_tails, dtype=np.int64)
        not_base = (
            ~np.isin(ordered_tails, base_tails)
            if n
            else np.zeros(0, dtype=np.bool_)
        )
        in_true = (
            np.isin(ordered_tails, true_tails)
            if n
            else np.zeros(0, dtype=np.bool_)
        )
        first_occurrence = np.zeros(n, dtype=np.bool_)
        if n:
            _, first_indices = np.unique(ordered_tails, return_index=True)
            first_occurrence[first_indices] = True
        new_unit = first_occurrence & not_base
        new_true_unit = new_unit & in_true
        cumulative_new = np.empty(n + 1, dtype=np.int64)
        cumulative_new_true = np.empty(n + 1, dtype=np.int64)
        cumulative_new[0] = 0
        cumulative_new_true[0] = 0
        np.cumsum(new_unit, dtype=np.int64, out=cumulative_new[1:])
        np.cumsum(new_true_unit, dtype=np.int64, out=cumulative_new_true[1:])

        if n == 0 or true_tails.size == 0:
            best_k, wrong = 0, 0
        else:
            base_true = int(np.isin(base_tails, true_tails).sum())
            best_k, wrong = _oracle_best_k(not_base, in_true, base_true)
        self.query_count += int(best_k > 0)
        self.selected += best_k
        self.selected_wrong += wrong
        self.new += int(cumulative_new[best_k])
        self.new_true += int(cumulative_new_true[best_k])
        return best_k

    def row(
        self,
        relation: int,
        truth_count: int,
        supply_count: int,
        queries: int,
    ) -> dict[str, int]:
        return {
            "relation": int(relation),
            "facts": int(truth_count),
            "supply": int(supply_count),
            "generated": int(supply_count + self.new),
            "missing": int(truth_count - supply_count - self.new_true),
            "extra": int(self.new - self.new_true),
            "queries": int(queries),
            "candidates": int(self.candidate_total),
            "selected": int(self.selected),
            "selected_wrong": int(self.selected_wrong),
        }


def _metrics(
    relation_rows: list[dict[str, int]],
    facts_n: int,
    policy_bytes: int,
) -> DecodeMetrics:
    supply = sum(int(row["supply"]) for row in relation_rows)
    generated = sum(int(row["generated"]) for row in relation_rows)
    missing = sum(int(row["missing"]) for row in relation_rows)
    extra = sum(int(row["extra"]) for row in relation_rows)
    evaluated = sum(int(row["facts"]) for row in relation_rows)
    untouched = max(0, int(facts_n) - evaluated)
    supply += untouched
    generated += untouched
    return DecodeMetrics(
        generated=int(generated),
        missing=int(missing),
        extra=int(extra),
        supply=int(supply),
        facts=int(facts_n),
        fact_rate=(supply + missing + extra) / max(float(facts_n), 1.0),
        threshold=-1.0,
        relation_rows=relation_rows,
        policy="query_topk_oracle",
        policy_bytes=int(policy_bytes),
    )


def _project_probe_rows(
    relation_rows: list[dict[str, int]],
    facts_n: int,
) -> DecodeMetrics:
    projected_missing = 0.0
    projected_extra = 0.0
    recalls: list[float] = []
    for row in relation_rows:
        deleted = max(int(row["facts"]) - int(row["supply"]), 0)
        recovered = max(deleted - int(row["missing"]), 0)
        queries = int(row["queries"])
        pool_size = int(row["pool_size"])
        if deleted == 0:
            recalls.append(1.0)
            continue
        if queries == 0:
            projected_missing += deleted
            recalls.append(0.0)
            continue
        scale = pool_size / queries
        projected_recovered = min(float(deleted), recovered * scale)
        projected_missing += deleted - projected_recovered
        projected_extra += int(row["selected_wrong"]) * scale
        recalls.append(projected_recovered / deleted)
    supply = sum(int(row["supply"]) for row in relation_rows)
    cost = supply + projected_missing + projected_extra
    return DecodeMetrics(
        generated=int(round(facts_n - projected_missing + projected_extra)),
        missing=int(round(projected_missing)),
        extra=int(round(projected_extra)),
        supply=int(supply),
        facts=int(facts_n),
        fact_rate=cost / max(float(facts_n), 1.0),
        threshold=-1.0,
        relation_rows=relation_rows,
        policy="query_topk_oracle_probe",
        projected_fact_rate=cost / max(float(facts_n), 1.0),
        projected_cost=cost,
        projected_recall_mean=sum(recalls) / max(len(recalls), 1),
        probe_meta={
            "queries_per_relation": 1024,
            "sampling": "uniform_packed_block",
            "ranking_only": True,
        },
    )


def decode_compression_probe(
    model: SupplyGatedDRUM,
    facts: np.ndarray,
    entity_count: int,
    relation_count: int,
    args: SimpleNamespace,
) -> DecodeMetrics:
    pass                                                                            
    model.eval()
    with torch.no_grad():
        hard_supply = model.weight.view(-1) >= model.hard_threshold
    hard_supply_np = hard_supply.detach().cpu().numpy().astype(np.bool_, copy=False)
    proof_graph, proof_supply = materialize_supply_proof_graph(
        facts, hard_supply, model.weight_param.device,
    )
    relation_index = build_relation_row_index(facts, list(range(int(relation_count))))
    rows: list[dict[str, int]] = []
    with hard_supply_context(model, proof_supply):
        with torch.no_grad():
            for relation in range(int(relation_count)):
                (
                    truth_count,
                    supply_count,
                    truth_by_head,
                    supply_by_head,
                    supply_heads,
                ) = relation_grouped_arrays(
                    facts,
                    relation,
                    hard_supply_np,
                    getattr(args, "_query_orientation_inverse_np", None),
                    precomputed_idx=relation_index[relation],
                )
                query_heads = query_heads_for_relation(
                    "supply_entities" if args.dataset == "family" else "supply_heads",
                    facts,
                    relation,
                    hard_supply_np,
                    getattr(args, "_query_orientation_inverse_np", None),
                    supply_heads,
                )
                pool_size = len(query_heads)
                if pool_size > 1024:
                    rng = np.random.default_rng(relation * 9176 + 17)
                    block = int(rng.integers(0, math.ceil(pool_size / 1024)))
                    query_heads = query_heads[block * 1024 : (block + 1) * 1024]
                accumulator = _OracleRelationAccumulator()
                for candidate in iter_direction_candidate_rows(
                    model,
                    relation,
                    query_heads,
                    truth_by_head,
                    supply_by_head,
                    proof_graph,
                    args,
                ):
                    accumulator.add_row(
                        candidate.ordered_tails,
                        candidate.true_tails,
                        candidate.base_tails,
                    )
                row = accumulator.row(relation, truth_count, supply_count, len(query_heads))
                row["truth_queries"] = len(truth_by_head)
                row["pool_size"] = pool_size
                rows.append(row)
    return _project_probe_rows(rows, fact_count(facts))


def _fact_keys(rows: np.ndarray, entity_count: int, relation_count: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
    return (rows[:, 0] * int(relation_count) + rows[:, 1]) * int(entity_count) + rows[:, 2]


def _unique_rows(rows: np.ndarray, entity_count: int, relation_count: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
    if rows.shape[0] == 0:
        return rows
    keys = _fact_keys(rows, entity_count, relation_count)
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    keep = np.ones(keys.shape[0], dtype=np.bool_)
    keep[1:] = keys[1:] != keys[:-1]
    return rows[order[keep]]


def _row_difference(
    left: np.ndarray,
    right: np.ndarray,
    entity_count: int,
    relation_count: int,
) -> np.ndarray:
    left = _unique_rows(left, entity_count, relation_count)
    right = _unique_rows(right, entity_count, relation_count)
    if left.shape[0] == 0 or right.shape[0] == 0:
        return left
    left_keys = _fact_keys(left, entity_count, relation_count)
    right_keys = _fact_keys(right, entity_count, relation_count)
    positions = np.searchsorted(right_keys, left_keys)
    present = positions < right_keys.shape[0]
    present[present] &= right_keys[positions[present]] == left_keys[present]
    return left[~present]


def _write_query_bundle(
    directory: str | Path,
    model: SupplyGatedDRUM,
    args: SimpleNamespace,
    facts: np.ndarray,
    support: np.ndarray,
    selected_parts: list[np.ndarray],
    policy_rows: list[tuple[int, int, int]],
    entity_count: int,
    relation_count: int,
) -> tuple[int, int, int]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    generated = _unique_rows(
        np.concatenate([support, *selected_parts], axis=0) if selected_parts else support,
        entity_count,
        relation_count,
    )
    truth = _unique_rows(facts, entity_count, relation_count)
    missing = _row_difference(truth, generated, entity_count, relation_count)
    extra = _row_difference(generated, truth, entity_count, relation_count)
    np.save(directory / "support.npy", support, allow_pickle=False)
    np.save(directory / "missing.npy", missing, allow_pickle=False)
    np.save(directory / "extra.npy", extra, allow_pickle=False)
    np.save(
        directory / "policy.npy",
        np.asarray(policy_rows, dtype=np.int64).reshape(-1, 3),
        allow_pickle=False,
    )
    torch.save(
        {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
            if key not in {"weight_param", "gate_grad_absmean_ema", "gate_grad_norm_steps"}
        },
        directory / "rules.pt",
    )
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "entity_count": int(entity_count),
                "relation_count": int(relation_count),
                "K": int(model.step),
                "L": int(model.rules),
                "support": int(support.shape[0]),
                "missing": int(missing.shape[0]),
                "extra": int(extra.shape[0]),
                "policy_rows": int(len(policy_rows)),
                "kernel_topk_edges": int(getattr(args, "kernel_topk_edges", 1 << 60)),
                "kernel_topk_nodes": int(getattr(args, "kernel_topk_nodes", 0)),
                "decode_batch_size": int(getattr(args, "decode_batch_size", 4096)),
                "decode_proof_microbatch_size": int(getattr(args, "decode_proof_microbatch_size", 0)),
                "decode_query_prune": str(getattr(args, "decode_query_prune", "none")),
                "decode_prune_threshold": float(getattr(args, "decode_query_rule_threshold", 0.0)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return int(support.shape[0]), int(missing.shape[0]), int(extra.shape[0])


def decode_query_topk_oracle(
    model: SupplyGatedDRUM,
    facts: np.ndarray,
    graph: Graph,
    entity_count: int,
    relation_count: int,
    args: SimpleNamespace,
    bundle_dir: str | Path | None = None,
) -> DecodeMetrics:
    pass                                                                      
    model.eval()
    with torch.no_grad():
        hard_supply = model.weight.view(-1) >= model.hard_threshold
    hard_supply_np = hard_supply.detach().cpu().numpy().astype(np.bool_, copy=False)
    support = np.asarray(facts[hard_supply_np], dtype=np.int64)
    if bool(getattr(args, "large_graph", False)):
        model._kernel_csr = None
        model._kernel_csr_key = None
        empty_long = torch.empty(0, dtype=torch.long, device=model.emb.device)
        empty_bool = torch.empty(0, dtype=torch.bool, device=model.emb.device)
        graph.head = graph.rel = graph.tail = empty_long
        graph.mask = empty_bool
        graph.mask_float = torch.empty(0, dtype=torch.float32, device=model.emb.device)
        graph.e2triple = (empty_long, None, empty_bool)
        graph.triple2e = (None, empty_long, empty_bool)
        graph.r2triple = (empty_long, None, empty_bool)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    proof_graph, proof_supply = materialize_supply_proof_graph(
        facts,
        hard_supply,
        model.weight_param.device,
    )
    orientation_inverse = getattr(args, "_query_orientation_inverse_np", None)
    query_source = str(args.threshold_qsth_query_source)
    relation_text = str(getattr(args, "decode_relations", ""))
    relation_ids = (
        [int(value) for value in relation_text.split(",") if value]
        if relation_text else list(range(int(relation_count)))
    )
    relation_index = build_relation_row_index(facts, relation_ids)
    relation_ids.sort(key=lambda r: int(relation_index[r].size))
    prepare_rule_first_hop_prune(model, relation_ids, facts, hard_supply_np, args)

    rows: list[dict[str, int]] = []
    query_count = 0
    max_candidate_count = 0
    policy_rows: list[tuple[int, int, int]] = []
    selected_parts: list[np.ndarray] = []
    print(
        json.dumps(
            {
                "event": "decode_relation_loop_start",
                "relations": len(relation_ids),
                "query_source": query_source,
                "max_queries": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    with hard_supply_context(model, proof_supply):
        with torch.no_grad():
            for index, relation in enumerate(relation_ids):
                relation_rows = relation_index[relation]
                if index < 3 or index + 1 == len(relation_ids):
                    print(
                        json.dumps(
                            {
                                "event": "decode_relation_progress",
                                "i": index + 1,
                                "n": len(relation_ids),
                                "relation": relation,
                                "rows": int(relation_rows.size),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                (
                    truth_count,
                    supply_count,
                    truth_by_head,
                    supply_by_head,
                    supply_heads,
                ) = relation_grouped_arrays(
                    facts,
                    relation,
                    hard_supply_np,
                    orientation_inverse,
                    precomputed_idx=relation_rows,
                )
                query_heads = query_heads_for_relation(
                    query_source,
                    facts,
                    relation,
                    hard_supply_np,
                    orientation_inverse,
                    supply_heads,
                )
                queries_before_prune = len(query_heads)
                query_heads = rule_first_hop_query_heads(relation, query_heads, args)
                accumulator = _OracleRelationAccumulator()
                for candidate in iter_direction_candidate_rows(
                    model,
                    relation,
                    query_heads,
                    truth_by_head,
                    supply_by_head,
                    proof_graph,
                    args,
                ):
                    best_k = accumulator.add_row(
                        candidate.ordered_tails,
                        candidate.true_tails,
                        candidate.base_tails,
                    )
                    if bundle_dir is not None and best_k > 0:
                        policy_rows.append((relation, candidate.query_head, best_k))
                        selected_parts.append(
                            np.column_stack(
                                (
                                    np.full(best_k, candidate.query_head, dtype=np.int64),
                                    np.full(best_k, relation, dtype=np.int64),
                                    candidate.ordered_tails[:best_k],
                                )
                            )
                        )
                row = accumulator.row(
                    relation,
                    truth_count,
                    supply_count,
                    len(query_heads),
                )
                row["truth_queries"] = len(truth_by_head)
                row["pool_size"] = len(query_heads)
                row["queries_before_prune"] = queries_before_prune
                rows.append(row)
                query_count += accumulator.query_count
                max_candidate_count = max(
                    max_candidate_count,
                    accumulator.max_candidate_count,
                )

    head_bits = max(1, int(max(0, entity_count - 1)).bit_length())
    k_bits = max(1, int(max(0, max_candidate_count)).bit_length())
    policy_bytes = int(math.ceil((head_bits + k_bits) * query_count / 8.0))
    metrics = _metrics(rows, fact_count(facts), policy_bytes)
    if bundle_dir is not None:
        support_n, missing_n, extra_n = _write_query_bundle(
            bundle_dir,
            model,
            args,
            facts,
            support,
            selected_parts,
            policy_rows,
            entity_count,
            relation_count,
        )
        if (support_n, missing_n, extra_n) != (metrics.supply, metrics.missing, metrics.extra):
            raise RuntimeError(
                "query bundle does not match decode metrics: "
                f"bundle={(support_n, missing_n, extra_n)} "
                f"metrics={(metrics.supply, metrics.missing, metrics.extra)}"
            )
    return metrics


def evaluate_threshold_policies(
    model: SupplyGatedDRUM,
    facts: np.ndarray,
    graph: Graph,
    entity_count: int,
    relation_count: int,
    args: SimpleNamespace,
    policies: list[str],
    threshold_sweep: str = "",
) -> list[DecodeMetrics]:
    del threshold_sweep
    unsupported = {str(policy) for policy in policies} - {"query_topk_oracle"}
    if unsupported:
        raise ValueError(f"unsupported decode policies: {sorted(unsupported)}")
    return [
        decode_query_topk_oracle(
            model,
            facts,
            graph,
            entity_count,
            relation_count,
            args,
        )
    ]
