"""
FGSM and PGD attacks (L_infinity) operating on [0,1] inputs.
Model is expected to do its own normalization internally (wrap with NormalizedModel).
"""
from typing import Optional
import torch
import torch.nn as nn


def _clamp(x, lower=0.0, upper=1.0):
    return torch.clamp(x, lower, upper)


def _grad_sign(loss, x):
    grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
    return grad.sign()


def fgsm_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    """Single-step FGSM under L_inf with epsilon in [0,1] scale.
    x, y must be on the same device as model.
    """
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = nn.CrossEntropyLoss()(logits, y)
    g = _grad_sign(loss, x_adv)
    x_adv = x_adv + eps * g
    x_adv = _clamp(x_adv)
    return x_adv.detach()


def pgd_linf(model: nn.Module, x: torch.Tensor, y: torch.Tensor, *, eps: float, alpha: float,
             steps: int, random_start: bool = True) -> torch.Tensor:
    """Projected Gradient Descent (L_inf) with optional random start.
    eps, alpha are in [0,1] scale.
    """
    model.eval()
    x = x.detach()
    if random_start:
        x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0., 1.)
    else:
        x_adv = x.clone()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = nn.CrossEntropyLoss()(logits, y)
        grad = _grad_sign(loss, x_adv)
        x_adv = x_adv.detach() + alpha * grad
        # Project back into Linf-ball and clip to [0,1]
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv = _clamp(x_adv)
    return x_adv.detach()
