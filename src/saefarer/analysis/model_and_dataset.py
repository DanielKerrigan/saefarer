from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from datasets import (
    Dataset,
    IterableDataset,
)
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from saefarer.analysis.types import ConfusionMatrix, ConfusionMatrixCell, ModelInfo

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from saefarer.analysis.config import AnalysisConfig


@torch.inference_mode()
def get_dataset_with_predictions(
    model: "PreTrainedModel",
    dataset: Dataset | IterableDataset | DataLoader,
    cfg: "AnalysisConfig",
) -> dict[str, torch.Tensor]:
    if isinstance(dataset, Dataset):
        ds = dataset[0 : cfg.total_analysis_sequences]
    else:
        if isinstance(dataset, IterableDataset):
            dataloader = DataLoader(
                dataset,  # type: ignore
                batch_size=cfg.total_analysis_sequences,
            )
        else:
            dataloader = DataLoader(
                dataset=dataset.dataset,
                shuffle=False,
                batch_size=cfg.total_analysis_sequences,
                collate_fn=dataset.collate_fn,
                num_workers=dataset.num_workers,
            )

        ds = next(iter(dataloader))

    predicted_probabilities = _get_model_predictions(model, ds, cfg)
    ds["pred_probs"] = predicted_probabilities
    ds["pred_label"] = predicted_probabilities.argmax(dim=1)

    return ds


@torch.inference_mode()
def _get_model_predictions(
    model: "PreTrainedModel",
    ds: dict[str, torch.Tensor],
    cfg: "AnalysisConfig",
) -> torch.Tensor:
    tokens = ds[cfg.tokens_column]
    attn_masks = ds[cfg.attn_mask_column]

    predicted_probabilities = torch.zeros(
        (tokens.shape[0], len(cfg.labels)),
        device=torch.device("cpu"),
        dtype=model.dtype,
    )

    offset = 0

    token_batches = tokens.split(cfg.model_batch_size_sequences)
    attn_mask_batches = attn_masks.split(cfg.model_batch_size_sequences)

    for token_batch, attn_mask_batch in zip(token_batches, attn_mask_batches):
        output = model(
            token_batch.to(cfg.device),
            attention_mask=attn_mask_batch.to(cfg.device),
        )
        probs = F.softmax(output.logits, dim=1)

        start = offset
        offset += probs.shape[0]
        end = offset

        predicted_probabilities[start:end, :] = probs.to("cpu")

    return predicted_probabilities


@torch.inference_mode()
def get_model_info(ds: dict[str, torch.Tensor], cfg: "AnalysisConfig") -> ModelInfo:
    label_indices = list(range(len(cfg.labels)))
    cm = get_confusion_matrix(ds["label"], ds["pred_label"], label_indices)

    mean_probabilities = ds["pred_probs"].mean(dim=0).tolist()

    return ModelInfo(
        labels=cfg.labels,
        label_indices=label_indices,
        cm=cm,
        mean_pred_label_probs=mean_probabilities,
    )


@torch.inference_mode()
def get_confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    label_indices: list[int],
) -> ConfusionMatrix:
    y_true_np = y_true.numpy(force=True)
    y_pred_np = y_pred.numpy(force=True)

    n_sequences = y_true_np.shape[0]

    error_count = (y_true_np != y_pred_np).sum().item()
    error_pct = error_count / n_sequences

    label_counts = np.bincount(y_true_np, minlength=len(label_indices))
    label_pcts = label_counts / n_sequences

    pred_label_counts = np.bincount(y_pred_np, minlength=len(label_indices))
    pred_label_pcts = pred_label_counts / n_sequences

    matrix = confusion_matrix(y_true=y_true_np, y_pred=y_pred_np, labels=label_indices)

    cells: list[ConfusionMatrixCell] = []

    for true_index in label_indices:
        for pred_index in label_indices:
            count = matrix[true_index, pred_index].item()
            pct = count / n_sequences
            cell = ConfusionMatrixCell(
                label=true_index,
                pred_label=pred_index,
                count=count,
                pct=pct,
            )
            cells.append(cell)

    correct_counts = np.diagonal(matrix)

    false_pos_counts = pred_label_counts - correct_counts
    false_pos_pcts = false_pos_counts / n_sequences

    false_neg_counts = label_counts - correct_counts
    false_neg_pcts = false_neg_counts / n_sequences

    cm = ConfusionMatrix(
        n_sequences=n_sequences,
        error_count=error_count,
        error_pct=error_pct,
        cells=cells,
        label_counts=label_counts.tolist(),
        label_pcts=label_pcts.tolist(),
        pred_label_counts=pred_label_counts.tolist(),
        pred_label_pcts=pred_label_pcts.tolist(),
        false_pos_counts=false_pos_counts.tolist(),
        false_pos_pcts=false_pos_pcts.tolist(),
        false_neg_counts=false_neg_counts.tolist(),
        false_neg_pcts=false_neg_pcts.tolist(),
    )

    return cm
