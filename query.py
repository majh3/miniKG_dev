                      


from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

try:
    from .decode_core import iter_direction_candidate_rows
    from .final_model import build_model
    from .graph import Graph
    from .proof import hard_supply_context
except ImportError:
    from decode_core import iter_direction_candidate_rows
    from final_model import build_model
    from graph import Graph
    from proof import hard_supply_context


def _index(rows: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    grouped: dict[tuple[int, int], list[int]] = {}
    for head, relation, tail in np.asarray(rows, dtype=np.int64).reshape(-1, 3):
        grouped.setdefault((int(head), int(relation)), []).append(int(tail))
    return {
        key: np.asarray(sorted(set(tails)), dtype=np.int64)
        for key, tails in grouped.items()
    }


class MiniKGQuery:
    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        self.entity_count = int(meta["entity_count"])
        self.relation_count = int(meta["relation_count"])
        self.dataset = str(meta["dataset"])
        support = np.load(directory / "support.npy", allow_pickle=False)
        self.context_heads = sorted(
            set(
                int(value)
                for value in (
                    np.concatenate((support[:, 0], support[:, 2]))
                    if self.dataset == "family" and support.size
                    else support[:, 0]
                ).tolist()
            )
        )
        self.support = _index(support)
        self.missing = _index(np.load(directory / "missing.npy", allow_pickle=False))
        self.extra = _index(np.load(directory / "extra.npy", allow_pickle=False))
        self.policy = {
            (int(relation), int(head)): int(k)
            for relation, head, k in np.load(directory / "policy.npy", allow_pickle=False).reshape(-1, 3)
        }

        device = torch.device("cuda")
        cfg = SimpleNamespace(
            dataset=self.dataset,
            K=int(meta["K"]),
            L=int(meta["L"]),
        )
        self.model = build_model(
            np.asarray(support, dtype=np.int64),
            self.entity_count,
            self.relation_count,
            cfg,
            device,
        )
        try:
            state = torch.load(directory / "rules.pt", map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(directory / "rules.pt", map_location=device)
        incompatible = self.model.load_state_dict(state, strict=False)
        if set(incompatible.missing_keys) != {
            "weight_param",
            "gate_grad_absmean_ema",
            "gate_grad_norm_steps",
        } or incompatible.unexpected_keys:
            raise ValueError(
                f"incompatible query model: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        self.model.eval()
        self.graph = Graph(torch.as_tensor(support, dtype=torch.long, device=device))
        self.graph_supply = torch.ones(len(support), dtype=torch.bool, device=device)
        self.args = SimpleNamespace(
            dataset=self.dataset,
            kernel_topk_edges=int(meta.get("kernel_topk_edges", (1 << 60) if self.dataset == "family" else 100)),
            kernel_topk_nodes=int(meta.get("kernel_topk_nodes", 0 if self.dataset == "family" else 100_000)),
            decode_batch_size=int(meta.get("decode_batch_size", 4096 if self.dataset == "family" else 1024)),
            decode_proof_microbatch_size=int(meta.get("decode_proof_microbatch_size", 0)),
            _decode_graph_compaction_allowed=False,
        )

    def project_many(self, heads: set[int], relation: int) -> dict[int, set[int]]:
        if relation < 0 or relation >= self.relation_count:
            raise ValueError(f"relation must be in [0, {self.relation_count})")
        heads = {int(head) for head in heads}
        answers = {
            head: set(self.support.get((head, relation), ()))
            | set(self.missing.get((head, relation), ()))
            for head in heads
        }

        requested_rule_heads = {head for head in heads if self.policy.get((relation, head), 0) > 0}
        if requested_rule_heads:
            with hard_supply_context(self.model, self.graph_supply):
                with torch.no_grad():
                    for row in iter_direction_candidate_rows(
                        self.model,
                        relation,
                        self.context_heads,
                        {},
                        {},
                        self.graph,
                        self.args,
                    ):
                        if row.query_head in requested_rule_heads:
                            answers[row.query_head].update(
                                row.ordered_tails[: self.policy[(relation, row.query_head)]].tolist()
                            )

        for head in heads:
            answers[head].difference_update(self.extra.get((head, relation), ()))
        return answers

    def project(self, heads: set[int], relation: int) -> set[int]:
        answers: set[int] = set()
        for tails in self.project_many(heads, relation).values():
            answers.update(tails)
        return answers

    def path(self, head: int, relations: list[int]) -> set[int]:
        answers = {int(head)}
        for relation in relations:
            answers = self.project(answers, int(relation))
            if not answers:
                break
        return answers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Query a finalized MiniKG representation")
    parser.add_argument("--dataset", required=True, choices=["family", "yago3-10"])
    parser.add_argument("--head", required=True, type=int)
    parser.add_argument("--relations", required=True, help="comma-separated relation IDs")
    ns = parser.parse_args(argv)
    relations = [int(value) for value in ns.relations.split(",") if value.strip()]
    if not relations:
        parser.error("--relations must contain at least one relation ID")
    answers = MiniKGQuery(Path("runs") / ns.dataset / "query").path(ns.head, relations)
    print(json.dumps({"count": len(answers), "answers": sorted(answers)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
