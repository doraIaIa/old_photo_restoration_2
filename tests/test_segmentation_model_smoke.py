from __future__ import annotations

import torch

from src.losses.segmentation import bce_dice_loss, tversky_loss
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


def test_tversky_loss_behaviour_for_match_and_miss() -> None:
    perfect_targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    perfect_logits = torch.tensor([[[[12.0, -12.0], [-12.0, 12.0]]]])
    missed_logits = torch.tensor([[[[-12.0, -12.0], [-12.0, -12.0]]]])

    perfect_loss = tversky_loss(perfect_logits, perfect_targets, alpha=0.3, beta=0.7)
    missed_loss = tversky_loss(missed_logits, perfect_targets, alpha=0.3, beta=0.7)

    assert perfect_loss.ndim == 0
    assert missed_loss.ndim == 0
    assert torch.isfinite(perfect_loss)
    assert torch.isfinite(missed_loss)
    assert float(perfect_loss) < 1e-3
    assert float(missed_loss) > float(perfect_loss)
