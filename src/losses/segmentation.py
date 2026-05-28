from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_shapes(logits: torch.Tensor, targets: torch.Tensor) -> None:
    if logits.shape != targets.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} phải khớp targets shape {tuple(targets.shape)}")


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    _validate_shapes(logits, targets)

    probabilities = torch.sigmoid(logits)
    targets = targets.float()

    probabilities = probabilities.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)

    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    dice_score = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice_score.mean()


def tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    _validate_shapes(logits, targets)
    if alpha < 0.0 or beta < 0.0:
        raise ValueError("tversky alpha và beta phải không âm.")

    probabilities = torch.sigmoid(logits)
    targets = targets.float()

    probabilities = probabilities.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)

    true_positive = (probabilities * targets).sum(dim=1)
    false_positive = (probabilities * (1.0 - targets)).sum(dim=1)
    false_negative = ((1.0 - probabilities) * targets).sum(dim=1)

    tversky_index = (true_positive + smooth) / (
        true_positive + alpha * false_positive + beta * false_negative + smooth
    )
    return 1.0 - tversky_index.mean()


def bce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    _validate_shapes(logits, targets)
    if bce_weight < 0.0 or dice_weight < 0.0:
        raise ValueError("bce_weight và dice_weight phải không âm.")
    if bce_weight == 0.0 and dice_weight == 0.0:
        raise ValueError("Không thể đặt cả bce_weight và dice_weight bằng 0.")

    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    return bce_weight * bce + dice_weight * dice


def bce_tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float = 0.5,
    tversky_weight: float = 0.5,
    alpha: float = 0.5,
    beta: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    _validate_shapes(logits, targets)
    if bce_weight < 0.0 or tversky_weight < 0.0:
        raise ValueError("bce_weight và tversky_weight phải không âm.")
    if bce_weight == 0.0 and tversky_weight == 0.0:
        raise ValueError("Không thể đặt cả bce_weight và tversky_weight bằng 0.")

    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    tversky = tversky_loss(logits, targets, alpha=alpha, beta=beta, smooth=smooth)
    return bce_weight * bce + tversky_weight * tversky
