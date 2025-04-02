from dataclasses import dataclass
from typing import TypedDict

import torch

# Aligned with types.ts


class Histogram(TypedDict):
    counts: list[int]
    thresholds: list[float]


class MarginalEffects(TypedDict):
    probs: list[list[float]]
    thresholds: list[float]


class DisplayToken(TypedDict):
    display: str
    token_ids: list[int]
    acts: list[float]
    max_act: float
    extras: dict[str, list[str]]
    is_special: bool


class FeatureTokenSequence(TypedDict):
    sequence_index: int
    display_tokens: list[DisplayToken]
    max_token_index: int
    label: int
    pred_label: int
    pred_probs: list[float]


class SequenceInterval(TypedDict):
    min_max_act: float
    max_max_act: float
    sequences: list[FeatureTokenSequence]


class ConfusionMatrixCell(TypedDict):
    label: int
    pred_label: int
    count: int
    pct: float


class ConfusionMatrix(TypedDict):
    n_sequences: int
    cells: list[ConfusionMatrixCell]
    label_counts: list[int]
    label_pcts: list[float]
    pred_label_counts: list[int]
    pred_label_pcts: list[float]
    false_pos_counts: list[int]
    false_pos_pcts: list[float]
    false_neg_counts: list[int]
    false_neg_pcts: list[float]


class FeatureData(TypedDict):
    sae_id: str
    feature_id: int
    max_act: float
    token_act_rate: float
    token_acts_histogram: Histogram
    sequence_act_rate: float
    sequence_acts_histogram: Histogram
    marginal_effects: MarginalEffects
    cm: ConfusionMatrix
    sequence_intervals: dict[str, SequenceInterval]
    mean_pred_label_probs: list[float]


class FeatureProjection(TypedDict):
    feature_ids: list[int]
    xs: list[float]
    ys: list[float]


class SAEData(TypedDict):
    sae_id: str
    num_total_features: int
    num_alive_features: int
    num_dead_features: int
    num_non_activating_features: int
    alive_feature_ids: list[int]
    token_act_rate_histogram: Histogram
    sequence_act_rate_histogram: Histogram
    feature_projection: FeatureProjection


class ModelInfo(TypedDict):
    labels: list[str]
    label_indices: list[int]
    cm: ConfusionMatrix
    mean_pred_label_probs: list[float]


# Python only


@dataclass
class SequenceIntervalIndices:
    min_max_act: float
    max_max_act: float
    indices: torch.Tensor
