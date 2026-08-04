                      


from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from data import load_facts
from query import MiniKGQuery


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["family", "yago3-10"])
    ns = parser.parse_args()
    started = time.perf_counter()
    facts = load_facts(f"data/{ns.dataset}/all_id.txt", "txt", assume_unique=False)
    truth: dict[tuple[int, int], set[int]] = defaultdict(set)
    for head, relation, tail in facts:
        truth[(int(head), int(relation))].add(int(tail))

    query = MiniKGQuery(Path("runs") / ns.dataset / "query")
    mismatches = []
    checked = 0
    for relation in range(query.relation_count):
        heads = {head for head, rel in truth if rel == relation}
        for index in (query.support, query.missing, query.extra):
            heads.update(head for head, rel in index if rel == relation)
        heads.update(head for rel, head in query.policy if rel == relation)
        actual = query.project_many(heads, relation)
        for head in heads:
            checked += 1
            expected = truth.get((head, relation), set())
            if actual[head] != expected:
                mismatches.append(
                    {
                        "head": head,
                        "relation": relation,
                        "missing": sorted(expected - actual[head])[:10],
                        "extra": sorted(actual[head] - expected)[:10],
                    }
                )
                if len(mismatches) >= 10:
                    break
        if mismatches:
            break

    paths = []
    if not mismatches:
        outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for (head, relation), tails in truth.items():
            for tail in sorted(tails):
                outgoing[head].append((relation, tail))
        for head in sorted(outgoing):
            for relation1, middle in outgoing[head]:
                if middle not in outgoing:
                    continue
                relation2 = outgoing[middle][0][0]
                expected = set()
                for first in truth.get((head, relation1), set()):
                    expected.update(truth.get((first, relation2), set()))
                actual = query.path(head, [relation1, relation2])
                paths.append(
                    {"head": head, "relations": [relation1, relation2], "exact": actual == expected}
                )
                if len(paths) == 2:
                    break
            if len(paths) == 2:
                break

    result = {
        "dataset": ns.dataset,
        "single_hop_queries": checked,
        "single_hop_mismatches": mismatches,
        "path_queries": paths,
        "exact": not mismatches and bool(paths) and all(row["exact"] for row in paths),
        "seconds": time.perf_counter() - started,
    }
    output = Path("runs") / ns.dataset / "query_verification.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
