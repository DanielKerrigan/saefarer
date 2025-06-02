"""
This is based on the SAE analysis code from sae_vis and SAEDashboard:
https://github.com/callummcdougall/sae_vis
https://github.com/jbloomAus/SAEDashboard
"""

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy import stats

from saefarer.analysis.model_and_dataset import get_confusion_matrix
from saefarer.analysis.types import (
    DisplayToken,
    FeatureData,
    FeatureTokenSequence,
    HistogramData,
    MarginalEffectsData,
    SequenceInterval,
    SequenceIntervalIndices,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from transformers import PreTrainedTokenizer

    from saefarer.analysis.config import AnalysisConfig
    from saefarer.analysis.types import DatasetInfo, ModelInfo


@torch.inference_mode()
def get_feature_data(
    feature_id: int,
    sae_id: str,
    dataset_info: "DatasetInfo",
    model_info: "ModelInfo",
    token_acts: torch.Tensor,
    positive_token_acts_mask: torch.Tensor,
    tokenizer: "PreTrainedTokenizer",
    ds: dict[str, torch.Tensor],
    cfg: "AnalysisConfig",
    rng: np.random.Generator,
) -> FeatureData:
    # Token activations
    positive_token_acts = token_acts[positive_token_acts_mask]
    token_act_rate = positive_token_acts.numel() / token_acts.numel()
    positive_token_acts_np = positive_token_acts.numpy(force=True)
    token_act_range = (
        positive_token_acts_np.min().item(),
        positive_token_acts_np.max().item(),
    )
    token_acts_histogram = _get_act_histogram(
        positive_token_acts_np, cfg.n_activation_bins, token_act_range
    )

    # Sequence activations
    sequence_acts = token_acts.max(dim=1)[0]
    positive_sequence_acts_mask = sequence_acts > 0
    positive_sequence_acts_mask_cpu = positive_sequence_acts_mask.cpu()
    positive_sequence_acts = sequence_acts[positive_sequence_acts_mask]
    sequence_act_rate = positive_sequence_acts.numel() / sequence_acts.numel()
    positive_sequence_acts_np = positive_sequence_acts.numpy(force=True)
    sequence_act_range = (
        positive_sequence_acts_np.min().item(),
        positive_sequence_acts_np.max().item(),
    )
    sequence_acts_histogram = _get_act_histogram(
        positive_sequence_acts_np, cfg.n_activation_bins, sequence_act_range
    )

    # Example sequences
    sequence_intervals = _get_example_sequences(
        feature_id,
        tokenizer,
        ds,
        token_acts,
        sequence_acts,
        sequence_acts_histogram["thresholds"],
        cfg,
        rng,
    )

    # Marginal effects
    marginal_effects = _get_sequence_level_marginal_effects(
        positive_sequence_acts,
        positive_sequence_acts_mask_cpu,
        sequence_acts_histogram["thresholds"],
        ds,
    )

    # Confusion matrix
    cm = get_confusion_matrix(
        ds["label"][positive_sequence_acts_mask_cpu],
        ds["pred_label"][positive_sequence_acts_mask_cpu],
        dataset_info["label_indices"],
    )

    # Additional feature statistics
    mean_pred_label_probs = (
        ds["pred_probs"][positive_sequence_acts_mask_cpu].mean(dim=0).tolist()
    )

    return FeatureData(
        sae_id=sae_id,
        feature_id=feature_id,
        max_act=token_acts.max().item(),
        token_act_rate=token_act_rate,
        token_acts_histogram=token_acts_histogram,
        sequence_act_rate=sequence_act_rate,
        sequence_acts_histogram=sequence_acts_histogram,
        marginal_effects=marginal_effects,
        cm=cm,
        sequence_intervals=sequence_intervals,
        mean_pred_label_probs=mean_pred_label_probs,
    )


@torch.inference_mode()
def _get_example_sequences(
    feature_index: int,
    tokenizer: "PreTrainedTokenizer",
    ds: dict[str, torch.Tensor],
    token_activations: torch.Tensor,
    sequence_activations: torch.Tensor,
    thresholds: list[float],
    cfg: "AnalysisConfig",
    rng: np.random.Generator,
) -> list[SequenceInterval]:
    interval_indices = _get_interval_indices(sequence_activations, thresholds, cfg, rng)

    sequence_intervals: list[SequenceInterval] = []

    for interval in interval_indices:
        key_seq: list[FeatureTokenSequence] = []

        for seq_i in interval.indices.tolist():
            tok_ids = ds[cfg.tokens_column][seq_i]
            acts = token_activations[seq_i]
            tok_i = int(torch.argmax(acts).item())

            token_sequence = _get_feature_token_sequence(
                feature_index=feature_index,
                tokenizer=tokenizer,
                input_ids=tok_ids.tolist(),
                activations=acts.tolist(),
                sequence_index=seq_i,
                token_index=tok_i,
                ds=ds,
                cfg=cfg,
            )
            key_seq.append(token_sequence)

        sequence_intervals.append(
            SequenceInterval(
                min_max_act=interval.min_max_act,
                max_max_act=interval.max_max_act,
                sequences=key_seq,
            )
        )

    return sequence_intervals


@torch.inference_mode()
def _get_interval_indices(
    sequence_activations: torch.Tensor,
    thresholds: list[float],
    cfg: "AnalysisConfig",
    rng: np.random.Generator,
) -> list[SequenceIntervalIndices]:
    sequence_indices: list[SequenceIntervalIndices] = []

    top_values, top_indices = torch.topk(
        sequence_activations, k=cfg.n_example_sequences
    )

    positive_mask = top_values > 0
    top_values_positive = top_values[positive_mask]
    top_indices_positive = top_indices[positive_mask]

    sequence_indices.append(
        SequenceIntervalIndices(
            min_max_act=top_values_positive.min().item(),
            max_max_act=top_values_positive.max().item(),
            indices=top_indices_positive,
        )
    )

    bins_per_interval = cfg.n_activation_bins // cfg.n_sequence_intervals
    interval_indices = list(range(0, len(thresholds), bins_per_interval))
    interval_thresholds = [thresholds[i] for i in interval_indices]
    interval_ranges = list(zip(interval_thresholds, interval_thresholds[1:]))

    for i, (interval_min, interval_max) in enumerate(interval_ranges):
        valid_indices = (
            (
                (sequence_activations >= interval_min)
                & (sequence_activations < interval_max)
            )
            .nonzero()
            .flatten()
        )

        if valid_indices.shape[0] > cfg.n_example_sequences:
            # https://stackoverflow.com/a/60564584
            rand_indices = torch.tensor(
                rng.choice(
                    valid_indices.shape[0],
                    cfg.n_example_sequences,
                    replace=False,
                )
            )

            valid_indices = valid_indices[rand_indices]

        sequence_indices.append(
            SequenceIntervalIndices(
                min_max_act=interval_min,
                max_max_act=interval_max,
                indices=valid_indices,
            )
        )

    return sequence_indices


@torch.inference_mode()
def _get_feature_token_sequence(
    feature_index: int,
    tokenizer: "PreTrainedTokenizer",
    input_ids: list[int],
    activations: list[float],
    sequence_index: int,
    token_index: int,
    ds: dict[str, torch.Tensor],
    cfg: "AnalysisConfig",
) -> FeatureTokenSequence:
    # token metadata columns

    token_extras: dict[str, list[str]] = {}

    for entry in cfg.extra_token_columns:
        if isinstance(entry, str):
            col, formatter = entry, str
        else:
            col, formatter = entry

        values = ds[col][sequence_index].tolist()
        token_extras[col] = [formatter(value) for value in values]

    display_tokens_subset, max_token_index = get_display_tokens(
        tokenizer=tokenizer,
        input_ids=input_ids,
        activations=activations,
        sequence_index=sequence_index,
        token_index=token_index,
        token_extras=token_extras,
        n_context_tokens=cfg.n_context_tokens,
    )

    # sequence metadata

    sequence_extras: dict[str, str] = {}

    for entry in cfg.extra_sequence_columns:
        if isinstance(entry, str):
            col, formatter = entry, str
        else:
            col, formatter = entry

        sequence_extras[col] = formatter(ds[col][sequence_index].item())

    token_sequence = FeatureTokenSequence(
        feature_index=feature_index,
        sequence_index=sequence_index,
        display_tokens=display_tokens_subset,
        max_token_index=max_token_index,
        label=int(ds["label"][sequence_index]),
        pred_label=int(ds["pred_label"][sequence_index]),
        pred_probs=ds["pred_probs"][sequence_index].tolist(),
        extras=sequence_extras,
    )

    return token_sequence


@torch.inference_mode()
def get_display_tokens(
    tokenizer: "PreTrainedTokenizer",
    input_ids: list[int],
    activations: list[float],
    sequence_index: int,
    token_index: int,
    token_extras: dict[str, list[str]],
    n_context_tokens: int,
):
    display_tokens: list[DisplayToken] = []

    seq = tokenizer.decode(input_ids)

    token_id_group = []
    activations_group = []
    extras_group = defaultdict(list)

    super_tokens = []

    max_super_token_index = -1

    for i in range(len(input_ids)):
        token_id_group.append(input_ids[i])
        activations_group.append(activations[i])
        for k, v in token_extras.items():
            extras_group[k].append(v[i])

        super_token = tokenizer.decode(token_id_group)

        if i == token_index:
            max_super_token_index = len(display_tokens)

        if seq.startswith("".join(super_tokens) + super_token):
            display_token = DisplayToken(
                display=super_token,
                token_ids=token_id_group,
                acts=activations_group,
                max_act=max(activations_group),
                extras=extras_group,
                is_special=super_token in tokenizer.all_special_tokens,
            )

            display_tokens.append(display_token)
            super_tokens.append(super_token)

            token_id_group = []
            activations_group = []
            extras_group = defaultdict(list)

    assert max_super_token_index != -1

    if token_id_group or activations_group:
        print(f"problems tokenizing {sequence_index}")

    # take a subset of the tokens

    if n_context_tokens >= 0:
        min_index = max(0, max_super_token_index - n_context_tokens)
        max_index = min(
            len(display_tokens) - 1, max_super_token_index + n_context_tokens
        )

        display_tokens_subset = display_tokens[min_index:max_index]
        max_token_index = max_super_token_index - min_index
    else:
        display_tokens_subset = display_tokens
        max_token_index = max_super_token_index

    return display_tokens_subset, max_token_index


@torch.inference_mode()
def _get_act_histogram(
    acts: "npt.NDArray[np.float64]",
    num_bins: int,
    acts_range: tuple[float, float] | None = None,
) -> HistogramData:
    hist_range = acts_range if acts_range is not None else (acts.min(), acts.max())
    hist, bin_edges = np.histogram(acts, bins=num_bins, range=hist_range)
    return HistogramData(counts=hist.tolist(), thresholds=bin_edges.tolist())


@torch.inference_mode()
def _get_sequence_level_marginal_effects(
    positive_acts: torch.Tensor,
    positive_acts_mask_cpu: torch.Tensor,
    bin_edges: list[float],
    ds: dict[str, torch.Tensor],
) -> MarginalEffectsData:
    pred_probs = ds["pred_probs"][positive_acts_mask_cpu]

    positive_acts_np = positive_acts.numpy(force=True)

    probabilities = []

    for i in range(pred_probs.shape[1]):
        statistic, _, _ = stats.binned_statistic(
            positive_acts_np,
            pred_probs[:, i],
            statistic="mean",
            bins=bin_edges,  # type: ignore
        )

        filled = np.nan_to_num(statistic, nan=-1).tolist()
        probabilities.append(filled)

    non_act_pred_probs = (
        ds["pred_probs"][~positive_acts_mask_cpu].mean(dim=0).nan_to_num(-1).tolist()
    )

    return MarginalEffectsData(
        probs=probabilities, thresholds=bin_edges, non_act_probs=non_act_pred_probs
    )
