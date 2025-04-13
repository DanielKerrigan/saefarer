import torch
from pytest import approx

from saefarer.analysis.model_and_dataset import get_confusion_matrix
from saefarer.analysis.types import ConfusionMatrix, ConfusionMatrixCell


def test_get_confusion_matrix():
    """
    This example is from Wikipedia
    https://en.wikipedia.org/wiki/Confusion_matrix
    """
    y_true = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    label_indices = [0, 1]

    expected_cm = ConfusionMatrix(
        n_sequences=12,
        error_count=3,
        error_pct=3 / 12,
        cells=[
            ConfusionMatrixCell(label=0, pred_label=0, count=3, pct=3 / 12),
            ConfusionMatrixCell(label=0, pred_label=1, count=1, pct=1 / 12),
            ConfusionMatrixCell(label=1, pred_label=0, count=2, pct=2 / 12),
            ConfusionMatrixCell(label=1, pred_label=1, count=6, pct=6 / 12),
        ],
        label_counts=[4, 8],
        label_pcts=[4 / 12, 8 / 12],
        pred_label_counts=[5, 7],
        pred_label_pcts=[5 / 12, 7 / 12],
        false_pos_counts=[2, 1],
        false_pos_pcts=[2 / 12, 1 / 12],
        false_neg_counts=[1, 2],
        false_neg_pcts=[1 / 12, 2 / 12],
    )

    actual_cm = get_confusion_matrix(
        y_true=y_true, y_pred=y_pred, label_indices=label_indices
    )

    assert expected_cm["n_sequences"] == actual_cm["n_sequences"]

    assert len(expected_cm["cells"]) == len(actual_cm["cells"])

    for expected_cell, actual_cell in zip(expected_cm["cells"], actual_cm["cells"]):
        assert expected_cell["label"] == actual_cell["label"]
        assert expected_cell["pred_label"] == actual_cell["pred_label"]
        assert expected_cell["count"] == actual_cell["count"]
        assert expected_cell["pct"] == approx(actual_cell["pct"])

    assert expected_cm["label_counts"] == actual_cm["label_counts"]
    assert expected_cm["label_pcts"] == approx(actual_cm["label_pcts"])
    assert expected_cm["pred_label_counts"] == actual_cm["pred_label_counts"]
    assert expected_cm["pred_label_pcts"] == approx(actual_cm["pred_label_pcts"])
    assert expected_cm["false_pos_counts"] == actual_cm["false_pos_counts"]
    assert expected_cm["false_pos_pcts"] == approx(actual_cm["false_pos_pcts"])
    assert expected_cm["false_neg_counts"] == actual_cm["false_neg_counts"]
    assert expected_cm["false_neg_pcts"] == approx(actual_cm["false_neg_pcts"])
