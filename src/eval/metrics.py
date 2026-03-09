"""Evaluation metrics for visual grounding."""

from __future__ import annotations


def compute_iou(
    pred: tuple[float, float, float, float],
    gt: tuple[float, float, float, float],
) -> float:
    """Compute Intersection over Union between two bboxes.

    Args:
        pred: Predicted bbox (x1, y1, x2, y2), normalized [0, 1].
        gt: Ground truth bbox (x1, y1, x2, y2), normalized [0, 1].

    Returns:
        IoU score in [0, 1].
    """
    x1 = max(pred[0], gt[0])
    y1 = max(pred[1], gt[1])
    x2 = min(pred[2], gt[2])
    y2 = min(pred[3], gt[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area_pred = (pred[2] - pred[0]) * (pred[3] - pred[1])
    area_gt = (gt[2] - gt[0]) * (gt[3] - gt[1])
    union = area_pred + area_gt - intersection

    if union == 0:
        return 0.0

    return intersection / union


def accuracy_at_threshold(
    predictions: list[tuple[float, float, float, float]],
    ground_truths: list[tuple[float, float, float, float]],
    threshold: float = 0.5,
) -> float:
    """Compute Acc@threshold — fraction of predictions with IoU >= threshold.

    Args:
        predictions: List of predicted bboxes.
        ground_truths: List of ground truth bboxes.
        threshold: IoU threshold (0.5 for Acc@0.5, 0.75 for Acc@0.75).

    Returns:
        Accuracy score in [0, 1].
    """
    assert len(predictions) == len(ground_truths)
    if not predictions:
        return 0.0

    correct = sum(
        compute_iou(pred, gt) >= threshold
        for pred, gt in zip(predictions, ground_truths)
    )
    return correct / len(predictions)


def mean_iou(
    predictions: list[tuple[float, float, float, float]],
    ground_truths: list[tuple[float, float, float, float]],
) -> float:
    """Compute mean IoU across all predictions."""
    assert len(predictions) == len(ground_truths)
    if not predictions:
        return 0.0

    ious = [compute_iou(p, g) for p, g in zip(predictions, ground_truths)]
    return sum(ious) / len(ious)
