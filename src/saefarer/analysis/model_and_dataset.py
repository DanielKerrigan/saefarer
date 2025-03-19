import torch
import torch.nn.functional as F
from datasets import (
    Dataset,
    IterableDataset,
)
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from saefarer.analysis.config import AnalysisConfig
from saefarer.analysis.types import ConfusionMatrixCell, ModelInfo


@torch.inference_mode()
def get_dataset_with_predictions(
    model: PreTrainedModel,
    dataset: Dataset | IterableDataset | DataLoader,
    cfg: AnalysisConfig,
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
    model: PreTrainedModel,
    ds: dict[str, torch.Tensor],
    cfg: AnalysisConfig,
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
def get_model_info(ds: dict[str, torch.Tensor], cfg: AnalysisConfig) -> ModelInfo:
    n_sequences = ds["label"].shape[0]
    label_indices = list(range(len(cfg.labels)))
    cm = _get_confusion_matrix(ds, label_indices)

    mean_probabilities = ds["pred_probs"].mean(dim=0).tolist()

    label_counts = ds["label"].unique(sorted=True, return_counts=True)[1].tolist()
    predicted_label_counts = (
        ds["pred_label"].unique(sorted=True, return_counts=True)[1].tolist()
    )

    return ModelInfo(
        n_sequences=n_sequences,
        labels=cfg.labels,
        label_indices=label_indices,
        cm=cm,
        mean_pred_label_probs=mean_probabilities,
        label_counts=label_counts,
        pred_label_counts=predicted_label_counts,
    )


@torch.inference_mode()
def _get_confusion_matrix(
    ds: dict[str, torch.Tensor], label_indices: list[int]
) -> list[ConfusionMatrixCell]:
    y_true = ds["label"].numpy(force=True)
    y_pred = ds["pred_label"].numpy(force=True)
    matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=label_indices)

    cells: list[ConfusionMatrixCell] = []

    for true_index in label_indices:
        for pred_index in label_indices:
            cells.append(
                ConfusionMatrixCell(
                    label=true_index,
                    pred_label=pred_index,
                    count=matrix[true_index, pred_index],
                )
            )

    return cells
