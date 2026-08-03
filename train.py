"""Champion training loop shared by Family and YAGO3-10."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

try:
    from .backend_args import build_backend_args
    from .data import build_truth_lookup as build_truth_by_query
    from .decode import evaluate_threshold_policies
    from .facts import encode_membership_keys, fact_count, facts_to_tensor, infer_entity_relation_count
    from .final_model import build_model, build_optimizers
    from .gate_grad_normalization import apply_balanced_global_travel
    from .graph import Graph
    from .loss import apply_chunked_nbe_gate_update, source_credit_aux_loss_final, target_net_benefit_loss_final
    from .metrics import DecodeMetrics, curve_record
    from .tnb_batching import tnb_batch_plan
except ImportError:
    from backend_args import build_backend_args
    from data import build_truth_lookup as build_truth_by_query
    from decode import evaluate_threshold_policies
    from facts import encode_membership_keys, fact_count, facts_to_tensor, infer_entity_relation_count
    from final_model import build_model, build_optimizers
    from gate_grad_normalization import apply_balanced_global_travel
    from graph import Graph
    from loss import apply_chunked_nbe_gate_update, source_credit_aux_loss_final, target_net_benefit_loss_final
    from metrics import DecodeMetrics, curve_record
    from tnb_batching import tnb_batch_plan


def append_curve_record(path_text: str, record: dict) -> None:
    if not path_text:
        return
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def seed_everything(cfg: SimpleNamespace, device: torch.device) -> torch.Generator:
    if cfg.dataset == "yago3-10":
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    return generator


def estimate_gate_stats(model, generator, n_facts: int, max_sample: int = 100_000) -> dict[str, float]:
    if n_facts <= max_sample:
        supply = model.weight.detach().view(-1)
    else:
        idx = torch.randint(0, n_facts, (max_sample,), generator=generator, device=model.weight_param.device)
        supply = torch.sigmoid(model.gate_logits(idx) / model.supply_temperature).detach().view(-1)
    return {
        "supply_mean": float(supply.mean().item()),
        "keep_rate": float((supply > 0.5).float().mean().item()),
        "mid_frac": float(((supply > 0.2) & (supply < 0.8)).float().mean().item()),
        "gate_p10": float(torch.quantile(supply.float(), 0.10).item()),
        "gate_p90": float(torch.quantile(supply.float(), 0.90).item()),
    }


def membership_keys(facts, facts_tensor, entity_count, relation_count, args, tag) -> torch.Tensor:
    print(json.dumps({"tag": tag, "event": "membership_keys_start", "facts": len(facts),
                      "membership_key_device": str(facts_tensor.device)}, sort_keys=True), flush=True)
    keys = torch.sort(encode_membership_keys(
        facts_tensor, entity_count, relation_count, "auto"
    )).values
    print(json.dumps({"tag": tag, "event": "membership_keys_done", "facts": len(facts),
                      "membership_key_count": int(keys.numel()),
                      "membership_key_device": str(keys.device)}, sort_keys=True), flush=True)
    return keys


def evaluate(model, facts, graph, entity_count, relation_count, args, cfg) -> list[DecodeMetrics]:
    return evaluate_threshold_policies(
        model, facts, graph, entity_count, relation_count, args,
        ["query_topk_oracle"], "",
    )


def configure_gate_learning_rate(model, optimizers, args, cfg, steps: int, n_facts: int, tag: str) -> None:
    travel = abs(math.log(0.8 / (1.0 - 0.8)) - math.log(0.05 / (1.0 - 0.05)))
    tau = float(model.supply_temperature)
    sparse_touches = (512 if cfg.dataset == "family" else 2048) * steps / max(float(n_facts), 1.0)
    # Both champion recipes use auto mode with pressure start at step zero.
    touches = max(float(steps), 1.0)
    gate_lr = cfg.kappa * travel * tau * tau / (touches * 0.25)
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            if any(parameter is model.weight_param for parameter in group["params"]):
                group["lr"] = gate_lr
    print(json.dumps({"tag": tag, "netbenefit_gate_lr": round(gate_lr, 4),
                      "expected_touches": round(touches, 2), "gate_lr_touch_mode": "auto",
                      "sparse_touches_raw": round(sparse_touches, 4),
                      "sparse_touches_clamped": round(max(sparse_touches, 1.0), 2),
                      "dense_active_steps": round(touches, 2)}), flush=True)
    gain = (0.25 / tau) * touches / (0.5 * max(float(steps) - 20, 1.0))
    args.netbenefit_st_pressure_gain = gain
    print(json.dumps({"tag": tag, "netbenefit_st_gain_auto": round(gain, 6)}), flush=True)


def train_once(facts: np.ndarray, orientation_inverse: np.ndarray, cfg: SimpleNamespace,
               steps: int, tag: str, collect_curve: bool = False):
    device = torch.device("cuda")
    entity_count, relation_count = infer_entity_relation_count(facts)
    args = build_backend_args(cfg, orientation_inverse, device)
    facts_tensor = facts_to_tensor(facts, device)
    graph = Graph(facts_tensor)
    truth_by_query = (build_truth_by_query(facts, orientation_inverse)
                      if cfg.dataset == "yago3-10" else {})
    generator = seed_everything(cfg, device)
    model = build_model(facts, entity_count, relation_count, cfg, device)
    optimizers = build_optimizers(model, cfg)
    n_facts = fact_count(facts)
    configure_gate_learning_rate(model, optimizers, args, cfg, steps, n_facts, tag)

    args._nbe_c_cache = torch.full((n_facts,), -1.0, dtype=torch.float32, device=device)
    args._nbe_c_cache_is_gate_local = False
    args._nbe_k_per_fact = None
    args._nbe_chunked_dense_update = False
    args._nbe_dense_update_pending = None
    train_idx = torch.arange(n_facts, dtype=torch.long, device=device)
    sorted_true_keys = (membership_keys(facts, facts_tensor, entity_count, relation_count, args, tag)
                        if cfg.dataset == "family"
                        else torch.empty((0,), dtype=torch.long, device=device))
    print(json.dumps({"tag": tag, "event": "train_pool", "facts": n_facts}, sort_keys=True), flush=True)

    curve: list[dict] = []
    best_state = None
    best_rate = float("inf")
    best_step = -1
    for step_idx in range(steps):
        model.train()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        args._nbe_dense_update_pending = None
        args._nbe_pressure_on = True
        args._nbe_anneal_frac = float(step_idx + 1) / max(steps, 1)
        plan = tnb_batch_plan(args, min(512 if cfg.dataset == "family" else 2048, n_facts))
        args._tnb_false_main_batch_count = plan.false_main_batch_count
        pos_local = torch.randint(0, n_facts, (plan.true_target_batch_size,),
                                  generator=generator, device=device)
        pos = train_idx[pos_local.long()]

        tnb_loss, tnb_true, tnb_false = target_net_benefit_loss_final(
            model, facts_tensor, pos, graph, truth_by_query, entity_count,
            relation_count, sorted_true_keys, args, generator,
        )
        aux_loss, aux_proof, aux_active = source_credit_aux_loss_final(
            model, facts_tensor, pos, graph, args,
        )
        pressure_loss = torch.zeros((), device=model.weight_param.device, dtype=model.weight_param.dtype)
        loss = tnb_loss + 0.5 * aux_loss + pressure_loss
        if loss.requires_grad:
            loss.backward()
        gate_stats = apply_balanced_global_travel(model, optimizers)
        gate_stats.update(apply_chunked_nbe_gate_update(model, args, optimizers))
        for optimizer in optimizers:
            optimizer.step()

        step = step_idx + 1
        if step % 100 == 0 or step == steps:
            with torch.no_grad():
                stats = estimate_gate_stats(model, generator, n_facts)
            record = {"tag": tag, "step": step, "true_main": plan.true_target_batch_size,
                      "false_main": plan.false_main_batch_count, "loss": float(loss.item()),
                      "tnb": float(tnb_loss.item()), "tnb_true": float(tnb_true.item()),
                      "tnb_false": float(tnb_false.item()), "source_credit_aux": float(aux_loss.item()),
                      "source_credit_aux_proof": float(aux_proof.item()),
                      "source_credit_aux_active": int(aux_active), **stats, **gate_stats}
            witness = getattr(args, "_nbe_w_cache", None)
            if witness is not None:
                record.update(witness_absmax=float(witness.abs().max().item()),
                              witness_absmean=float(witness.abs().mean().item()))
            print(json.dumps(record, sort_keys=True), flush=True)

        if collect_curve and cfg.dataset == "family" and (step % 400 == 0 or step == steps):
            step_metrics = evaluate(model, facts, graph, entity_count, relation_count, args, cfg)
            record = curve_record(tag, step, step_metrics[0])
            curve.append(record)
            append_curve_record(f"runs/{cfg.dataset}/curve.jsonl", record)
            print(json.dumps({"tag": tag, "curve_decode": record}, sort_keys=True), flush=True)
            if step_metrics[0].fact_rate < best_rate:
                best_rate, best_step = step_metrics[0].fact_rate, step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                print(json.dumps({"tag": tag, "best_checkpoint": {"step": step,
                                  "fact_rate": best_rate}}, sort_keys=True), flush=True)

        if step % (400 if cfg.dataset == "family" else 1000) == 0 or step == steps:
            directory = Path(f"runs/{cfg.dataset}/snapshots")
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"model_step{step}.pt"
            torch.save(model.state_dict(), path)
            print(json.dumps({"tag": tag, "snapshot": str(path), "step": step}), flush=True)

    if best_state is not None and best_step != steps:
        model.load_state_dict(best_state)
        print(json.dumps({"tag": tag, "best_checkpoint_restored": {
            "step": best_step, "fact_rate": best_rate}}, sort_keys=True), flush=True)

    args._decode_graph_compaction_allowed = True
    args._nbe_c_cache = None
    args._nbe_w_cache = None
    args._nbe_jitter_cache = None
    args._nbe_k_per_fact = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    all_metrics = evaluate(model, facts, graph, entity_count, relation_count, args, cfg)
    metrics = all_metrics[0]
    args._last_decode_metrics = all_metrics
    if collect_curve and not any(row["step"] == steps for row in curve):
        record = curve_record(tag, steps, metrics)
        append_curve_record(f"runs/{cfg.dataset}/curve.jsonl", record)
        print(json.dumps({"tag": tag, "curve_decode": record}, sort_keys=True), flush=True)
    print(json.dumps({"tag": tag, "decode": asdict(metrics),
                      "decode_compare": [asdict(item) for item in all_metrics]}, sort_keys=True), flush=True)
    return model, args, metrics
