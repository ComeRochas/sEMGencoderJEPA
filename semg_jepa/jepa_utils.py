from __future__ import annotations

import torch
import torch.nn.functional as F


def update_ema(student, teacher, momentum: float):
    with torch.no_grad():
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)


def variance_regularizer(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=(0, 1), unbiased=False) + eps)
    return torch.mean(F.relu(1 - std))


def embedding_std_mean(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=(0, 1), unbiased=False) + eps)
    return std.mean()
