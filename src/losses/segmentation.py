from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} phải khớp targets shape {tuple(targets.shape)}")

    probabilities = torch.sigmoid(logits)
    targets = targets.float()

    probabilities = probabilities.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)

    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    dice_score = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice_score.mean()


def bce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} phải khớp targets shape {tuple(targets.shape)}")

    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    return bce_weight * bce + dice_weight * dice
