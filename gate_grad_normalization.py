"""Balance-preserving global normalization for supply-gate gradients.

The complete gate gradient is multiplied by one scalar after backward.  This
keeps every sign and every pressure/credit ratio unchanged while removing the
graph-, batch-, and witness-dependent magnitude from the SGD update.
"""

from __future__ import annotations

import math
import torch


MODE = "balanced_global_travel"
_EMA_BUFFER = "gate_grad_absmean_ema"
_STEP_BUFFER = "gate_grad_norm_steps"


def register_state(model: torch.nn.Module) -> None:
    """Register persistent normalization state."""
    if not hasattr(model, _EMA_BUFFER):
        model.register_buffer(_EMA_BUFFER, torch.zeros((), dtype=torch.float64), persistent=True)
    if not hasattr(model, _STEP_BUFFER):
        model.register_buffer(_STEP_BUFFER, torch.zeros((), dtype=torch.long), persistent=True)


def _gate_lr(model: torch.nn.Module, optimizers: list[torch.optim.Optimizer]) -> float:
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            if any(param is model.weight_param for param in group["params"]):
                return float(group["lr"])
    raise RuntimeError("no optimizer owns model.weight_param")


@torch.no_grad()
def _dense_abs_stats(gradient: torch.Tensor, chunk_size: int = 4_000_000) -> tuple[float, float, int]:
    flat = gradient.detach().view(-1)
    abs_sum = 0.0
    abs_max = 0.0
    nonzero = 0
    for start in range(0, int(flat.numel()), int(chunk_size)):
        chunk = flat[start : start + int(chunk_size)]
        if not bool(torch.isfinite(chunk).all().item()):
            raise FloatingPointError("non-finite supply-gate gradient")
        nonzero += int(torch.count_nonzero(chunk).item())
        abs_chunk = chunk.abs()
        abs_sum += float(abs_chunk.sum(dtype=torch.float64).item())
        if int(abs_chunk.numel()) > 0:
            abs_max = max(abs_max, float(abs_chunk.max().item()))
    return abs_sum, abs_max, nonzero


@torch.no_grad()
def _gradient_abs_stats(gradient: torch.Tensor) -> tuple[float, float, int]:
    if gradient.is_sparse:
        values = gradient.coalesce().values().detach()
        if not bool(torch.isfinite(values).all().item()):
            raise FloatingPointError("non-finite sparse supply-gate gradient")
        nonzero = int(torch.count_nonzero(values).item())
        if nonzero == 0:
            return 0.0, 0.0, 0
        abs_values = values.abs()
        return (
            float(abs_values.sum(dtype=torch.float64).item()),
            float(abs_values.max().item()),
            nonzero,
        )
    return _dense_abs_stats(gradient)


@torch.no_grad()
def apply_balanced_global_travel(
    model: torch.nn.Module,
    optimizers: list[torch.optim.Optimizer],
) -> dict[str, int | float | str]:
    """Scale the accumulated gate gradient by one common scalar.

    The target is expressed in actual logit displacement, so an upstream
    auto-lr or a different base SGD lr cannot silently change the travel dose.
    """
    register_state(model)
    gradient = model.weight_param.grad
    if gradient is None:
        return {
            "gate_grad_normalization": MODE,
            "gate_grad_nnz": 0,
            "gate_grad_scale": 0.0,
            "gate_logit_step_absmean": 0.0,
            "gate_logit_step_absmax": 0.0,
        }

    abs_sum, abs_max, nonzero = _gradient_abs_stats(gradient)
    if nonzero == 0:
        return {
            "gate_grad_normalization": MODE,
            "gate_grad_nnz": 0,
            "gate_grad_scale": 0.0,
            "gate_logit_step_absmean": 0.0,
            "gate_logit_step_absmax": 0.0,
        }

    abs_mean = abs_sum / float(nonzero)
    steps = int(getattr(model, _STEP_BUFFER).item())
    ema_old = float(getattr(model, _EMA_BUFFER).item())
    ema = abs_mean if steps == 0 else 0.9 * ema_old + (1.0 - 0.9) * abs_mean
    getattr(model, _EMA_BUFFER).fill_(ema)
    getattr(model, _STEP_BUFFER).add_(1)

    lr = _gate_lr(model, optimizers)
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError(f"gate optimizer lr must be finite and positive, got {lr}")
    scale = min(
        0.02 / max(lr * ema, 1e-30),
        (0.2 if model.hard_threshold == 0.3 else 1_000_000.0) / max(lr * abs_max, 1e-30),
    )
    if not math.isfinite(scale) or scale < 0.0:
        raise FloatingPointError(f"invalid gate-gradient scale {scale}")

    if gradient.is_sparse:
        gradient._values().mul_(scale)
    else:
        gradient.mul_(scale)
    return {
        "gate_grad_normalization": MODE,
        "gate_grad_nnz": int(nonzero),
        "gate_grad_absmean_raw": float(abs_mean),
        "gate_grad_absmax_raw": float(abs_max),
        "gate_grad_absmean_ema": float(ema),
        "gate_grad_scale": float(scale),
        "gate_optimizer_lr": float(lr),
        "gate_logit_step_absmean": float(lr * scale * abs_mean),
        "gate_logit_step_absmax": float(lr * scale * abs_max),
    }
