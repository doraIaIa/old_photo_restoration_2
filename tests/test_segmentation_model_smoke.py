from __future__ import annotations

import torch

from src.losses.segmentation import bce_dice_loss
from src.models.segmenter import CrackSegmenter
from src.utils.metrics import binary_f1, binary_iou, binary_precision, binary_recall


def test_segmentation_model_loss_and_metrics_smoke() -> None:
    model = CrackSegmenter()
    inputs = torch.randn(2, 3, 512, 512)
    targets = torch.randint(0, 2, (2, 1, 512, 512)).float()

    logits = model(inputs)
    loss = bce_dice_loss(logits, targets)

    assert logits.shape == (2, 1, 512, 512)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    iou = binary_iou(logits, targets)
    f1 = binary_f1(logits, targets)
    precision = binary_precision(logits, targets)
    recall = binary_recall(logits, targets)

    for metric in (iou, f1, precision, recall):
        assert metric.ndim == 0
        assert torch.isfinite(metric)
        assert float(metric) >= 0.0
        assert float(metric) <= 1.0
