from __future__ import annotations

import torch
import torch.nn.functional as F


def update_ema(student, teacher, momentum: float):
    with torch.no_grad():
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)


def update_ema_encoder(student_encoder, teacher_encoder, momentum: float):
    """EMA update for encoder params + buffers.

    Floating-point parameters are EMA updated.
    Floating buffers (e.g. BatchNorm running stats) are EMA updated.
    Non-floating buffers are copied exactly.
    """

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


def variance_regularizer(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=(0, 1), unbiased=False) + eps)
    return torch.mean(F.relu(1 - std))


def embedding_std_mean(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=(0, 1), unbiased=False) + eps)
    return std.mean()
