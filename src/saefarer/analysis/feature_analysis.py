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
from saefarer.utils import top_k_indices_values

if TYPE_CHECKING:
    import numpy.typing as npt
    from transformers import PreTrainedTokenizer

    from saefarer.analysis.config import AnalysisConfig
    from saefarer.analysis.types import ModelInfo


@torch.inference_mode()
def get_feature_data(
    feature_id: int,
    sae_id: str,
    model_info: "ModelInfo",
    token_acts: torch.Tensor,
    positive_token_acts_mask: torch.Tensor,
    tokenizer: "PreTrainedTokenizer",
    ds: dict[str, torch.Tensor],
    cfg: "AnalysisConfig",
    rng: np.random.Generator,
) -> FeatureData:
    # Sequence activations
    sequence_acts = token_acts.max(dim=1)[0]
    positive_sequence_acts_mask = sequence_acts > 0
    positive_sequence_acts_mask_cpu = positive_sequence_acts_mask.cpu()
    positive_sequence_acts = sequence_acts[positive_sequence_acts_mask]
    sequence_act_rate = positive_sequence_acts.numel() / sequence_acts.numel()
    positive_sequence_acts_np = positive_sequence_acts.numpy(force=True)
    sequence_acts_histogram = _get_act_histogram(
        positive_sequence_acts_np, cfg.n_activation_bins
    )

    # Token activations
    positive_token_acts = token_acts[positive_token_acts_mask]
    token_act_rate = positive_token_acts.numel() / token_acts.numel()
    positive_token_acts_np = positive_token_acts.numpy(force=True)
    token_acts_histogram = _get_act_histogram(
        positive_token_acts_np, cfg.n_activation_bins
    )

    # Example sequences
    sequence_intervals = _get_example_sequences(tokenizer, ds, token_acts, cfg, rng)

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
        model_info["label_indices"],
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
    tokenizer: "PreTrainedTokenizer",
    ds: dict[str, torch.Tensor],
    feature_activations: torch.Tensor,
    cfg: "AnalysisConfig",
    rng: np.random.Generator,
) -> dict[str, SequenceInterval]:
    interval_indices = _get_interval_indices(feature_activations, cfg, rng)

    sequence_intervals: dict[str, SequenceInterval] = {}

    for key, interval in interval_indices.items():
        key_seq: list[FeatureTokenSequence] = []

        for point in interval.indices:
            seq_i = int(point[0].item())

            tok_ids = ds[cfg.tokens_column][seq_i]
            acts = feature_activations[seq_i]

            extras: dict[str, list[str]] = {}

            for entry in cfg.extra_token_columns:
                if isinstance(entry, str):
                    col, formatter = entry, str
                else:
                    col, formatter = entry

                values = ds[col][seq_i]

                # this would be the case if the value is the same for
                # every token in the sequence
                if values.dim() == 0:
                    values = [values.item()] * tok_ids.shape[0]
                else:
                    values = values.tolist()

                extras[col] = [formatter(value) for value in values]

            token_sequence = _get_feature_token_sequence(
                tokenizer=tokenizer,
                input_ids=tok_ids.tolist(),
                activations=acts.tolist(),
                extras=extras,
                sequence_index=seq_i,
                ds=ds,
                cfg=cfg,
            )
            key_seq.append(token_sequence)

        sequence_intervals[key] = SequenceInterval(
            min_max_act=interval.min_max_act,
            max_max_act=interval.max_max_act,
            sequences=key_seq,
        )

    return sequence_intervals


@torch.inference_mode()
def _get_interval_indices(
    feature_activations: torch.Tensor,
    cfg: "AnalysisConfig",
    rng: np.random.Generator,
) -> dict[str, SequenceIntervalIndices]:
    sequence_indices: dict[str, SequenceIntervalIndices] = {}

    top_indices, top_values = top_k_indices_values(
        feature_activations, k=cfg.n_example_sequences, largest=True
    )

    sequence_indices["Max Activations"] = SequenceIntervalIndices(
        top_values.min().item(),
        top_values.max().item(),
        top_indices,
    )

    activation_ranges = torch.linspace(
        feature_activations.min(),
        feature_activations.max(),
        cfg.n_sequence_intervals + 1,
    )

    interval_min_max = reversed(list(zip(activation_ranges, activation_ranges[1:])))

    for i, (interval_min, interval_max) in enumerate(interval_min_max):
        valid_indices = torch.stack(
            torch.where(
                (feature_activations >= interval_min)
                & (feature_activations < interval_max)
            ),
            dim=-1,
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

        sequence_indices[f"Interval {i + 1}"] = SequenceIntervalIndices(
            interval_min.item(),
            interval_max.item(),
            valid_indices,
        )

    return sequence_indices


@torch.inference_mode()
def _get_feature_token_sequence(
    tokenizer: "PreTrainedTokenizer",
    input_ids: list[int],
    activations: list[float],
    extras: dict[str, list[str]],
    sequence_index: int,
    ds: dict[str, torch.Tensor],
    cfg: "AnalysisConfig",
) -> FeatureTokenSequence:
    display_tokens: list[DisplayToken] = []

    seq = tokenizer.decode(input_ids)

    token_id_group = []
    activations_group = []
    extras_group = defaultdict(list)

    cleaned_tokens = []

    for i in range(len(input_ids)):
        token_id_group.append(input_ids[i])
        activations_group.append(activations[i])
        for k, v in extras.items():
            extras_group[k].append(v[i])

        clean_token = tokenizer.decode(token_id_group)

        if seq.startswith("".join(cleaned_tokens) + clean_token):
            display_token = DisplayToken(
                display=clean_token,
                token_ids=token_id_group,
                acts=activations_group,
                max_act=max(activations_group),
                extras=extras_group,
                is_special=clean_token in tokenizer.all_special_tokens,
            )

            display_tokens.append(display_token)
            cleaned_tokens.append(clean_token)

            token_id_group = []
            activations_group = []
            extras_group = defaultdict(list)

    max_super_token_index = np.argmax([x["max_act"] for x in display_tokens]).item()

    if cfg.n_context_tokens >= 0:
        min_index = max(0, max_super_token_index - cfg.n_context_tokens)
        max_index = min(
            len(display_tokens) - 1, max_super_token_index + cfg.n_context_tokens
        )

        display_tokens_subset = display_tokens[min_index:max_index]
        max_token_index = max_super_token_index - min_index
    else:
        display_tokens_subset = display_tokens
        max_token_index = max_super_token_index

    token_sequence = FeatureTokenSequence(
        sequence_index=sequence_index,
        display_tokens=display_tokens_subset,
        max_token_index=max_token_index,
        label=int(ds["label"][sequence_index]),
        pred_label=int(ds["pred_label"][sequence_index]),
        pred_probs=ds["pred_probs"][sequence_index].tolist(),
    )

    return token_sequence


@torch.inference_mode()
def _get_act_histogram(acts: "npt.NDArray[np.float64]", num_bins: int) -> HistogramData:
    acts_range = (0, acts.max())
    hist, bin_edges = np.histogram(acts, bins=num_bins, range=acts_range)
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
