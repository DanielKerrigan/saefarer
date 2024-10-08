from typing import Dict, List, TypedDict


class Histogram(TypedDict):
    counts: List[int]
    thresholds: List[float]


class FeatureTokenSequence(TypedDict):
    token: List[str]
    activation: List[float]
    extras: Dict[str, List[str]]
    max_index: int


class CumSumPercentL1Norm(TypedDict):
    n_neurons: List[int]
    cum_sum: List[float]


class CumSumPercentL1NormRange(TypedDict):
    mins: List[float]
    maxs: List[float]


class SequenceInterval(TypedDict):
    min_activation: float
    max_activation: float
    sequences: List[FeatureTokenSequence]


class FeatureData(TypedDict):
    sae_id: str
    feature_id: int
    activation_rate: float
    max_activation: float
    n_neurons_majority_l1_norm: int
    cumsum_percent_l1_norm: CumSumPercentL1Norm
    activations_histogram: Histogram
    sequence_intervals: Dict[str, SequenceInterval]


class FeatureProjection(TypedDict):
    feature_id: List[int]
    x: List[float]
    y: List[float]


class SAEData(TypedDict):
    sae_id: str
    num_total_features: int
    num_alive_features: int
    num_dead_features: int
    num_non_activating_features: int
    alive_feature_ids: List[int]
    activation_rate_histogram: Histogram
    dimensionality_histogram: Histogram
    cumsum_percent_l1_norm_range: CumSumPercentL1NormRange
    feature_projection: FeatureProjection


class LogData(TypedDict):
    elapsed_seconds: float
    n_training_batches: int
    n_training_tokens: int
    loss: float
    mse_loss: float
    aux_loss: float
    n_dead_features: int
    mean_n_batches_since_fired: float
    max_n_batches_since_fired: int
