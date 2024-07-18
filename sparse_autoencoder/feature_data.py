from typing import Dict, List, TypedDict


class Histogram(TypedDict):
    counts: List[int]
    thresholds: List[float]


class TokenSequence(TypedDict):
    tokens: List[str]
    activation: List[float]
    max_index: int


class FeatureData(TypedDict):
    firing_rate: float
    activations_histogram: Histogram
    sequences: List[TokenSequence]


class SAEData(TypedDict):
    num_dead_features: int
    num_alive_features: int
    firing_rate_histogram: Histogram
    dead_feature_indices: List[int]
    alive_feature_indices: List[int]
    feature_index_to_path: Dict[int, str]
