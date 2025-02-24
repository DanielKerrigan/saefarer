"""
This is based on the SAE analysis code from sae_vis and SAEDashboard:
https://github.com/callummcdougall/sae_vis
https://github.com/jbloomAus/SAEDashboard
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import stats
from transformers import PreTrainedTokenizer

from saefarer.config import AnalysisConfig
from saefarer.model import SAE
from saefarer.types import (
    DisplayToken,
    FeatureData,
    FeatureTokenSequence,
    Histogram,
    MarginalEffects,
    SequenceInterval,
)
from saefarer.utils import (
    freedman_diaconis_torch,
    top_k_indices_values,
    torch_histogram,
)


@torch.inference_mode()
def get_feature_data(
    feature_id: int,
    sae_id: str,
    sae: SAE,
    feature_activations: torch.Tensor,
    positive_activation_mask: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    ds: Dict[str, torch.Tensor],
    cfg: AnalysisConfig,
    rng: np.random.Generator,
) -> FeatureData:
    positive_activations = feature_activations[positive_activation_mask]

    sequence_intervals = _get_sequence_data(
        tokenizer, ds, feature_activations, positive_activations, cfg, rng
    )

    activation_rate = positive_activations.numel() / feature_activations.numel()

    activations_histogram = _get_activation_histogram(positive_activations)
    marginal_effects = _get_marginal_effects(
        positive_activations, positive_activation_mask, ds
    )

    return FeatureData(
        sae_id=sae_id,
        feature_id=feature_id,
        activation_rate=activation_rate,
        max_activation=feature_activations.max().item(),
        activations_histogram=activations_histogram,
        marginal_effects=marginal_effects,
        sequence_intervals=sequence_intervals,
    )


@torch.inference_mode()
def _get_sequence_indices(
    feature_activations: torch.Tensor,
    positive_activations: torch.Tensor,
    cfg: AnalysisConfig,
    rng: np.random.Generator,
) -> Dict[str, Tuple[float, float, torch.Tensor]]:
    sequence_indices: Dict[str, Tuple[float, float, torch.Tensor]] = {}

    top_indices, top_values = top_k_indices_values(
        feature_activations, k=cfg.n_example_sequences, largest=True
    )

    sequence_indices["Max Activations"] = (
        top_values.min().item(),
        top_values.max().item(),
        top_indices,
    )

    min_act = positive_activations.min()
    max_act = positive_activations.max()

    activation_ranges = torch.linspace(min_act, max_act, cfg.n_sequence_intervals + 1)

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

        sequence_indices[f"Interval {i + 1}"] = (
            interval_min.item(),
            interval_max.item(),
            valid_indices,
        )

    return sequence_indices


@torch.inference_mode()
def _get_sequence_data(
    tokenizer: PreTrainedTokenizer,
    ds: Dict[str, torch.Tensor],
    feature_activations: torch.Tensor,
    positive_activations: torch.Tensor,
    cfg: AnalysisConfig,
    rng: np.random.Generator,
) -> Dict[str, SequenceInterval]:
    sequence_indices = _get_sequence_indices(
        feature_activations, positive_activations, cfg, rng
    )

    sequence_intervals: Dict[str, SequenceInterval] = {}

    for key, (interval_min, interval_max, indices) in sequence_indices.items():
        key_seq: List[FeatureTokenSequence] = []

        for point in indices:
            seq_i = int(point[0].item())
            tok_i = int(point[1].item())

            # min_tok_i = max(0, tok_i - cfg.n_context_tokens)
            # max_tok_i = min(cfg.model_sequence_length, tok_i + cfg.n_context_tokens)

            min_tok_i = 0
            max_tok_i = cfg.model_sequence_length - 1

            tok_ids = ds[cfg.dataset_column][seq_i, min_tok_i : max_tok_i + 1]
            acts = feature_activations[seq_i, min_tok_i : max_tok_i + 1]

            extras: Dict[str, List[str]] = {}

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
                    values = values[min_tok_i : max_tok_i + 1].tolist()

                extras[col] = [formatter(value) for value in values]

            token_sequence = _get_feature_token_sequence(
                tokenizer=tokenizer,
                input_ids=tok_ids,
                activations=acts,
                extras=extras,
            )
            key_seq.append(token_sequence)

        sequence_intervals[key] = SequenceInterval(
            min_activation=interval_min, max_activation=interval_max, sequences=key_seq
        )

    return sequence_intervals


@torch.inference_mode()
def _get_feature_token_sequence(
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    activations: torch.Tensor,
    extras: Dict[str, List[str]],
) -> FeatureTokenSequence:
    display_tokens: List[DisplayToken] = []

    seq = tokenizer.decode(input_ids)

    token_id_group = []
    activations_group = []
    extras_group = defaultdict(list)

    cleaned_tokens = []

    for i in range(input_ids.shape[0]):
        token_id_group.append(input_ids[i])
        activations_group.append(activations[i])
        for k, v in extras.items():
            extras_group[k].append(v[i])

        clean_token = tokenizer.decode(token_id_group)

        if seq.startswith("".join(cleaned_tokens) + clean_token):
            display_token = DisplayToken(
                display=clean_token,
                token_ids=token_id_group,
                activations=activations_group,
                max_activation=max(activations_group),
                extras=extras_group,
            )

            display_tokens.append(display_token)
            cleaned_tokens.append(clean_token)

            token_id_group = []
            activations_group = []
            extras_group = defaultdict(list)

    token_sequence = FeatureTokenSequence(
        display_tokens=display_tokens,
        max_index=np.argmax([x["max_activation"] for x in display_tokens]).item(),
    )

    return token_sequence


@torch.inference_mode()
def _get_activation_histogram(
    positive_activations: torch.Tensor,
) -> Histogram:
    num_bins = min(freedman_diaconis_torch(positive_activations), 64)
    counts, thresholds = torch_histogram(positive_activations, bins=num_bins)
    return Histogram(counts=counts.tolist(), thresholds=thresholds.tolist())


@torch.inference_mode()
def _get_marginal_effects(
    positive_activations: torch.Tensor,
    positive_activation_mask: torch.Tensor,
    ds: Dict[str, torch.Tensor],
) -> MarginalEffects:
    num_bins = min(freedman_diaconis_torch(positive_activations), 64)
    positive_activations_numpy = positive_activations.numpy(force=True)
    bin_edges = np.histogram_bin_edges(positive_activations_numpy, num_bins)

    n_tokens, n_classes = ds["predicted_probabilities"].shape

    predictions_reshaped = (
        ds["predicted_probabilities"]
        .unsqueeze(1)
        .expand((n_tokens, positive_activation_mask.shape[1], n_classes))
    )

    positive_predictions_numpy = predictions_reshaped[
        positive_activation_mask.to("cpu")
    ].numpy(force=True)

    probabilities = []

    for i in range(n_classes):
        statistic, _, _ = stats.binned_statistic(
            positive_activations_numpy,
            positive_predictions_numpy[:, i],
            statistic="mean",
            bins=bin_edges,
        )

        probabilities.append(statistic.tolist())

    return MarginalEffects(probabilities=probabilities, thresholds=bin_edges.tolist())
