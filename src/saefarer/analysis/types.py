from dataclasses import dataclass
from typing import Literal, TypedDict

import torch

# Aligned with types.ts


class HistogramData(TypedDict):
    counts: list[int]
    thresholds: list[float]


class MarginalEffectsData(TypedDict):
    probs: list[list[float]]
    thresholds: list[float]
    non_act_probs: list[float]


class DisplayToken(TypedDict):
    display: str
    token_ids: list[int]
    acts: list[float]
    max_act: float
    extras: dict[str, list[str]]
    is_special: bool


class FeatureTokenSequence(TypedDict):
    feature_index: int
    sequence_index: int
    display_tokens: list[DisplayToken]
    max_token_index: int
    label: int
    pred_label: int
    pred_probs: list[float]
    extras: dict[str, str]


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
    error_count: int
    error_pct: float
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
    token_acts_histogram: HistogramData
    sequence_act_rate: float
    sequence_acts_histogram: HistogramData
    marginal_effects: MarginalEffectsData
    cm: ConfusionMatrix
    sequence_intervals: list[SequenceInterval]
    mean_pred_label_probs: list[float]


class FeatureProjection(TypedDict):
    feature_ids: list[int]
    xs: list[float]
    ys: list[float]


class SAEData(TypedDict):
    sae_id: str
    n_total_features: int
    n_alive_features: int
    n_dead_features: int
    n_non_activating_features: int
    alive_feature_ids: list[int]
    token_act_rate_histogram: HistogramData
    sequence_act_rate_histogram: HistogramData
    feature_projection: FeatureProjection


class FeatureIdRankingOption(TypedDict):
    kind: Literal["feature_id"]
    descending: bool


class SequenceActRateRankingOption(TypedDict):
    kind: Literal["sequence_act_rate"]
    descending: bool


class LabelRankingOption(TypedDict):
    kind: Literal["label"]
    true_label: str
    pred_label: str
    descending: bool


RankingOption = (
    FeatureIdRankingOption | SequenceActRateRankingOption | LabelRankingOption
)


class DatasetInfo(TypedDict):
    labels: list[str]
    label_indices: list[int]
    n_sequences: int
    n_tokens: int


class ModelInfo(TypedDict):
    cm: ConfusionMatrix
    mean_pred_label_probs: list[float]
    log_loss: float


class InferenceInput(TypedDict):
    feature_index: int
    sequence: str


# Python only


@dataclass
class SequenceIntervalIndices:
    min_max_act: float
    max_max_act: float
    indices: torch.Tensor
