from typing import Dict, List, TypedDict


class Histogram(TypedDict):
    counts: List[int]
    thresholds: List[float]


class MarginalEffects(TypedDict):
    probabilities: List[List[float]]
    thresholds: List[float]


class DisplayToken(TypedDict):
    display: str
    token_ids: List[int]
    activations: List[float]
    max_activation: float
    extras: Dict[str, List[str]]


class FeatureTokenSequence(TypedDict):
    display_tokens: List[DisplayToken]
    max_index: int


class SequenceInterval(TypedDict):
    min_activation: float
    max_activation: float
    sequences: List[FeatureTokenSequence]


class FeatureData(TypedDict):
    sae_id: str
    feature_id: int
    activation_rate: float
    max_activation: float
    activations_histogram: Histogram
    marginal_effects: MarginalEffects
    sequence_intervals: Dict[str, SequenceInterval]


class FeatureProjection(TypedDict):
    feature_ids: List[int]
    xs: List[float]
    ys: List[float]


class SAEData(TypedDict):
    sae_id: str
    num_total_features: int
    num_alive_features: int
    num_dead_features: int
    num_non_activating_features: int
    alive_feature_ids: List[int]
    activation_rate_histogram: Histogram
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
