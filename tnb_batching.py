"""Split the nominal training batch into true and false targets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TnbBatchPlan:
    true_target_batch_size: int
    false_main_batch_count: int


def tnb_batch_plan(args, nominal_batch_size: int) -> TnbBatchPlan:
    nominal = max(1, int(nominal_batch_size))
    false_count = 0 if args.dataset == "family" else min(nominal - 1, int(round(nominal * 0.5)))
    return TnbBatchPlan(
        true_target_batch_size=max(1, nominal - false_count),
        false_main_batch_count=false_count,
    )
