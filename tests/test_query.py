from __future__ import annotations

import unittest
from contextlib import nullcontext
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from decode import _project_probe_rows, _row_difference, _write_query_bundle
from decode_core import DirectionCandidateRow, rule_first_hop_query_heads
from backend_args import build_backend_args
from config import parse_config
from metrics import DecodeMetrics
from query import MiniKGQuery
from relation_profile import select_relations
from train import _advance_early_stop


class QueryTest(unittest.TestCase):
    def test_large_graph_features_are_enabled_by_dataset(self) -> None:
        for dataset, edges, nodes in (("freebase", 50, 50_000), ("wikidata5m", 100, 0)):
            cfg = parse_config(["--dataset", dataset])
            args = build_backend_args(cfg, np.zeros(2, dtype=np.bool_), torch.device("cpu"))
            self.assertTrue(args.large_graph)
            self.assertEqual((args.kernel_topk_edges, args.kernel_topk_nodes), (edges, nodes))
            self.assertEqual(args.decode_query_prune, "rule_first_hop")
            self.assertEqual(args.decode_query_rule_threshold, 0.2)
        cfg = parse_config(["--dataset", "family"])
        args = build_backend_args(cfg, np.zeros(2, dtype=np.bool_), torch.device("cpu"))
        self.assertFalse(args.large_graph)
        self.assertEqual(args.decode_query_prune, "none")

    def test_decode_prune_removes_unreachable_heads(self) -> None:
        args = SimpleNamespace(
            decode_query_prune="rule_first_hop",
            _decode_query_active_channels={3: [1, 5]},
            _decode_query_channel_heads={
                1: np.asarray([2, 7]),
                5: np.asarray([9]),
            },
        )
        self.assertEqual(
            rule_first_hop_query_heads(3, [1, 2, 7, 8, 9], args),
            [2, 7, 9],
        )

    def test_relation_profile_uses_public_threshold(self) -> None:
        rows = [
            {"relation": 0, "facts": 10, "supply": 6, "missing": 1, "extra": 0},
            {"relation": 1, "facts": 10, "supply": 8, "missing": 0, "extra": 0},
        ]
        metrics = DecodeMetrics(0, 0, 0, 0, 20, 0.0, -1.0, rows)
        self.assertEqual(select_relations(metrics, 0.8), [0])

    def test_probe_projection(self) -> None:
        metrics = _project_probe_rows(
            [{
                "relation": 0,
                "facts": 100,
                "supply": 40,
                "missing": 50,
                "extra": 2,
                "queries": 10,
                "pool_size": 20,
                "selected_wrong": 2,
                "generated": 52,
                "candidates": 0,
                "selected": 12,
            }],
            100,
        )
        self.assertEqual((metrics.supply, metrics.missing, metrics.extra), (40, 40, 4))
        self.assertAlmostEqual(metrics.fact_rate, 0.84)

    def test_early_stop_uses_warmup_and_patience(self) -> None:
        material, stale, stop = _advance_early_stop("yago3-10", 1000, 0.8, float("inf"), 0)
        self.assertEqual((material, stale, stop), (0.8, 0, False))
        for step in (6000, 7000, 8000, 9000):
            material, stale, stop = _advance_early_stop("yago3-10", step, 0.8, material, stale)
            self.assertFalse(stop)
        material, stale, stop = _advance_early_stop("yago3-10", 10000, 0.8, material, stale)
        self.assertTrue(stop)

    def test_row_difference(self) -> None:
        left = np.asarray([[0, 0, 1], [0, 0, 2], [1, 0, 2]], dtype=np.int64)
        right = np.asarray([[0, 0, 2]], dtype=np.int64)
        self.assertEqual(
            {tuple(row) for row in _row_difference(left, right, 3, 1).tolist()},
            {(0, 0, 1), (1, 0, 2)},
        )

    def test_corrected_path_without_rule_output(self) -> None:
        query = MiniKGQuery.__new__(MiniKGQuery)
        query.relation_count = 2
        query.support = {
            (0, 0): np.asarray([1], dtype=np.int64),
            (1, 1): np.asarray([3], dtype=np.int64),
        }
        query.missing = {
            (0, 0): np.asarray([2], dtype=np.int64),
            (2, 1): np.asarray([4], dtype=np.int64),
        }
        query.extra = {(0, 0): np.asarray([9], dtype=np.int64)}
        query.policy = {}
        query.context_heads = [0, 1, 2]
        self.assertEqual(query.path(0, [0]), {1, 2})
        self.assertEqual(query.path(0, [0, 1]), {3, 4})

    def test_rule_output_is_corrected(self) -> None:
        query = MiniKGQuery.__new__(MiniKGQuery)
        query.relation_count = 1
        query.support = {(0, 0): np.asarray([1], dtype=np.int64)}
        query.missing = {}
        query.extra = {(0, 0): np.asarray([9], dtype=np.int64)}
        query.policy = {(0, 0): 2}
        query.context_heads = [0, 5]
        query.model = object()
        query.graph = object()
        query.graph_supply = object()
        query.args = SimpleNamespace()
        row = DirectionCandidateRow(
            relation=0,
            query_head=0,
            ordered_tails=np.asarray([9, 2], dtype=np.int64),
            ordered_scores=np.asarray([1.0, 0.5]),
            true_tails=np.empty(0, dtype=np.int64),
            base_tails=np.asarray([1], dtype=np.int64),
        )
        with mock.patch("query.hard_supply_context", return_value=nullcontext()), mock.patch(
            "query.iter_direction_candidate_rows", side_effect=lambda *args: iter([row])
        ):
            self.assertEqual(query.project_many({0, 7}, 0), {0: {1, 2}, 7: set()})
            self.assertEqual(query.project({0}, 0), {1, 2})

    def test_bundle_stores_exact_corrections(self) -> None:
        class FakeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.step = 2
                self.rules = 16
                self.weight_param = torch.nn.Parameter(torch.zeros(2, 1))
                self.rule = torch.nn.Parameter(torch.ones(1))

        truth = np.asarray([[0, 0, 1], [0, 0, 2]], dtype=np.int64)
        support = np.asarray([[0, 0, 1]], dtype=np.int64)
        selected = np.asarray([[0, 0, 9], [0, 0, 2]], dtype=np.int64)
        with TemporaryDirectory() as temporary:
            counts = _write_query_bundle(
                temporary,
                FakeModel(),
                SimpleNamespace(dataset="family"),
                truth,
                support,
                [selected],
                [(0, 0, 2)],
                10,
                1,
            )
            self.assertEqual(counts, (1, 0, 1))
            root = Path(temporary)
            self.assertEqual(np.load(root / "missing.npy").shape, (0, 3))
            self.assertEqual(np.load(root / "extra.npy").tolist(), [[0, 0, 9]])
            self.assertEqual(json.loads((root / "meta.json").read_text())["policy_rows"], 1)
            self.assertGreater((root / "rules.pt").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
