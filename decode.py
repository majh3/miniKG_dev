"""Single-direction, uncapped query-top-k decode used by both champions."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import torch

try:
    from .decode_core import (
        build_relation_row_index,
        iter_direction_candidate_rows,
        materialize_supply_proof_graph,
        query_heads_for_relation,
        relation_grouped_arrays,
    )
    from .drum import SupplyGatedDRUM
    from .facts import fact_count
    from .graph import Graph
    from .metrics import DecodeMetrics
    from .proof import hard_supply_context
except ImportError:  # direct script execution
    from decode_core import (
        build_relation_row_index,
        iter_direction_candidate_rows,
        materialize_supply_proof_graph,
        query_heads_for_relation,
        relation_grouped_arrays,
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
    """Choose the first prefix maximizing gain, recovered truth, then errors."""
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
    """Count one relation without materializing decoded triple sets."""

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
    ) -> None:
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


def decode_query_topk_oracle(
    model: SupplyGatedDRUM,
    facts: np.ndarray,
    graph: Graph,
    entity_count: int,
    relation_count: int,
    args: SimpleNamespace,
) -> DecodeMetrics:
    """Run the exact uncapped decoder used for the reported champion rates."""
    model.eval()
    with torch.no_grad():
        hard_supply = model.weight.view(-1) >= model.hard_threshold
    hard_supply_np = hard_supply.detach().cpu().numpy().astype(np.bool_, copy=False)
    proof_graph, proof_supply = materialize_supply_proof_graph(
        facts,
        hard_supply,
        model.weight_param.device,
    )
    orientation_inverse = getattr(args, "_query_orientation_inverse_np", None)
    query_source = "supply_entities" if args.dataset == "family" else "supply_heads"
    relation_ids = list(range(int(relation_count)))
    relation_index = build_relation_row_index(facts, relation_ids)
    relation_ids.sort(key=lambda r: int(relation_index[r].size))

    rows: list[dict[str, int]] = []
    query_count = 0
    max_candidate_count = 0
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
                row = accumulator.row(
                    relation,
                    truth_count,
                    supply_count,
                    len(query_heads),
                )
                row["truth_queries"] = len(truth_by_head)
                row["pool_size"] = len(query_heads)
                rows.append(row)
                query_count += accumulator.query_count
                max_candidate_count = max(
                    max_candidate_count,
                    accumulator.max_candidate_count,
                )

    head_bits = max(1, int(max(0, entity_count - 1)).bit_length())
    k_bits = max(1, int(max(0, max_candidate_count)).bit_length())
    policy_bytes = int(math.ceil((head_bits + k_bits) * query_count / 8.0))
    return _metrics(rows, fact_count(facts), policy_bytes)


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
