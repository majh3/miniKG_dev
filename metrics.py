"""Small value objects used by the final training path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DecodeMetrics:
    generated: int
    missing: int
    extra: int
    supply: int
    facts: int
    fact_rate: float
    threshold: float
    relation_rows: list[dict[str, int]]
    policy: str = "global_threshold"
    policy_bytes: int = 0
    # L1 probe projection (ranking-only; absent for formal uncapped)
    projected_fact_rate: float | None = None
    projected_cost: float | None = None
    projected_recall_mean: float | None = None
    probe_meta: dict[str, Any] = field(default_factory=dict)


def curve_record(tag: str, step: int, metrics: DecodeMetrics) -> dict[str, int | float | str]:
    rec: dict[str, int | float | str | list | dict | None] = {
        "tag": tag,
        "step": int(step),
        "facts": int(metrics.facts),
        "supply": int(metrics.supply),
        "missing": int(metrics.missing),
        "extra": int(metrics.extra),
        "generated": int(metrics.generated),
        "fact_rate": float(metrics.fact_rate),
        "threshold": float(metrics.threshold),
        "policy": str(metrics.policy),
        "policy_bytes": int(metrics.policy_bytes),
        "relation_rows": list(metrics.relation_rows),
    }
    if metrics.projected_fact_rate is not None:
        rec["projected_fact_rate"] = float(metrics.projected_fact_rate)
    if metrics.projected_cost is not None:
        rec["projected_cost"] = float(metrics.projected_cost)
    if metrics.projected_recall_mean is not None:
        rec["projected_recall_mean"] = float(metrics.projected_recall_mean)
    if metrics.probe_meta:
        rec["probe_meta"] = dict(metrics.probe_meta)
    return rec
