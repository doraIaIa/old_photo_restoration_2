from __future__ import annotations

import torch


def _prepare_binary_tensors(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    from_logits: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if preds.shape != targets.shape:
        raise ValueError(f"preds shape {tuple(preds.shape)} phải khớp targets shape {tuple(targets.shape)}")

    predictions = torch.sigmoid(preds) if from_logits else preds
    predictions = (predictions >= threshold).float()
    targets = (targets >= 0.5).float()
    return predictions, targets


def binary_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    from_logits: bool = True,
) -> torch.Tensor:
    """Tính IoU cho segmentation nhị phân. `preds` có thể là logits hoặc probabilities."""
    predictions, targets = _prepare_binary_tensors(preds, targets, threshold, from_logits)
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    return (intersection + eps) / (union + eps)


def binary_f1(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    from_logits: bool = True,
) -> torch.Tensor:
    """Tính F1 cho segmentation nhị phân. `preds` có thể là logits hoặc probabilities."""
    predictions, targets = _prepare_binary_tensors(preds, targets, threshold, from_logits)
    tp = (predictions * targets).sum()
    fp = (predictions * (1.0 - targets)).sum()
    fn = ((1.0 - predictions) * targets).sum()
    return (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)


def binary_precision(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    from_logits: bool = True,
) -> torch.Tensor:
    """Tính precision cho segmentation nhị phân. `preds` có thể là logits hoặc probabilities."""
    predictions, targets = _prepare_binary_tensors(preds, targets, threshold, from_logits)
    tp = (predictions * targets).sum()
    fp = (predictions * (1.0 - targets)).sum()
    return (tp + eps) / (tp + fp + eps)


def binary_recall(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    from_logits: bool = True,
) -> torch.Tensor:
    """Tính recall cho segmentation nhị phân. `preds` có thể là logits hoặc probabilities."""
    predictions, targets = _prepare_binary_tensors(preds, targets, threshold, from_logits)
    tp = (predictions * targets).sum()
    fn = ((1.0 - predictions) * targets).sum()
    return (tp + eps) / (tp + fn + eps)
