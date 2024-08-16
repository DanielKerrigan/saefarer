from typing import Dict, List, TypedDict


class Histogram(TypedDict):
    counts: List[int]
    thresholds: List[float]


class TokenSequence(TypedDict):
    token: List[str]
    activation: List[float]
    max_index: int


class CumSumPercentL1Norm(TypedDict):
    n_neurons: List[int]
    cum_sum: List[float]


class FeatureData(TypedDict):
    sae_id: str
    feature_id: int
    activation_rate: float
    max_activation: float
    n_neurons_majority_l1_norm: int
    cumsum_percent_l1_norm: CumSumPercentL1Norm
    activations_histogram: Histogram
    sequences: Dict[str, List[TokenSequence]]


class FeatureProjection(TypedDict):
    feature_id: List[int]
    x: List[float]
    y: List[float]


class SAEData(TypedDict):
    sae_id: str
    num_alive_features: int
    num_dead_features: int
    alive_feature_ids: List[int]
    dead_feature_ids: List[int]
    activation_rate_histogram: Histogram
    dimensionality_histogram: Histogram
    feature_projection: FeatureProjection
