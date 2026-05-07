"""JEPA pretraining utilities.

Loss design (anti-collapse, VICReg-style hybrid)
------------------------------------------------
The previous setup applied F.normalize before a cosine-similarity loss with a
variance-hinge regularizer. That combination was structurally broken: L2-
normalizing a D-dim vector caps each per-feature std at ~1/sqrt(D), so the
hinge `relu(1 - std)` was always saturated near 1 regardless of collapse,
providing no anti-collapse gradient. Loss plateaued at exactly 1.0 (cosine ~ 0
+ saturated hinge ~ 1), the textbook collapse signature.

The new objective drops L2 normalization and computes three regularizers on
LayerNorm-stabilized features:

- invariance: smooth-L1 between LayerNorm(student-pred) and LayerNorm(teacher-target).
              I-JEPA / data2vec-2 style. Robust to outliers, well-scaled.
- variance:   VICReg hinge `mean(relu(target_std - std(z, dim=0)))` where std is
              taken over (B*T) tokens per feature dim. Now meaningful: with
              LayerNorm targets, per-dim std lives on a scale where target_std=1
              is reachable, so the hinge actively pushes against collapse.
- covariance: VICReg off-diagonal squared covariance, normalized by D. Pushes
              feature dimensions toward statistical decorrelation, fighting
              "informational collapse" where dims become redundant.

Default weights (inv=25, var=25, cov=1) follow VICReg. The asymmetric BYOL
structure (predictor on student, EMA teacher with no gradient) is kept — it
provides additional stability on top of the explicit anti-collapse terms.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def update_ema(student, teacher, momentum: float):
    with torch.no_grad():
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)


def update_ema_encoder(student_encoder, teacher_encoder, momentum: float):
    """EMA update for encoder params + floating buffers; non-floating buffers copied."""
    with torch.no_grad():
        s_params = dict(student_encoder.named_parameters())
        t_params = dict(teacher_encoder.named_parameters())
        for name, t_param in t_params.items():
            s_param = s_params[name]
            if torch.is_floating_point(t_param):
                t_param.data.mul_(momentum).add_(s_param.data, alpha=1 - momentum)
            else:
                t_param.data.copy_(s_param.data)

        s_buffers = dict(student_encoder.named_buffers())
        t_buffers = dict(teacher_encoder.named_buffers())
        for name, t_buffer in t_buffers.items():
            s_buffer = s_buffers[name]
            if torch.is_floating_point(t_buffer):
                t_buffer.data.mul_(momentum).add_(s_buffer.data, alpha=1 - momentum)
            else:
                t_buffer.data.copy_(s_buffer.data)


def variance_regularizer(z: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """VICReg variance hinge: per-feature std across tokens should reach target_std.

    z: [N, D] (caller flattens batch + time dims).
    """
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return torch.mean(F.relu(target_std - std))


def covariance_regularizer(z: torch.Tensor) -> torch.Tensor:
    """VICReg covariance term: sum of squared off-diagonal entries of cov(z), normalized by D.

    z: [N, D].
    """
    n, d = z.shape
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / max(1, n - 1)
    off = cov - torch.diag(torch.diag(cov))
    return (off ** 2).sum() / d


def feature_std_stats(z: torch.Tensor, eps: float = 1e-4):
    """Returns (mean_std, min_std) — diagnostics for collapse monitoring.

    Healthy training: mean_std → ~1.0, min_std > 0.
    Collapse: min_std → 0 (some dim is dead).
    """
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return std.mean(), std.min()


def cosine_momentum_schedule(step: int, total_steps: int, base: float, final: float) -> float:
    """Cosine schedule from `base` → `final` over `total_steps`. Bounded at the endpoints."""
    if total_steps <= 0:
        return final
    progress = min(1.0, max(0.0, step / total_steps))
    return final - (final - base) * 0.5 * (1.0 + math.cos(math.pi * progress))


def warmup_cosine_lr(step: int, warmup_steps: int, total_steps: int,
                     base_lr: float, min_lr: float = 0.0) -> float:
    """Linear warmup over `warmup_steps`, then cosine decay to `min_lr` by `total_steps`."""
    if step < warmup_steps and warmup_steps > 0:
        return base_lr * (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
