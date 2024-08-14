from typing import List, TypedDict


class Histogram(TypedDict):
    counts: List[int]
    thresholds: List[float]


class TokenSequence(TypedDict):
    token: List[str]
    activation: List[float]
    max_index: int


class FeatureData(TypedDict):
    feature_id: int
    sae_id: str
    firing_rate: float
    activations_histogram: Histogram
    sequences: List[TokenSequence]


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
    firing_rate_histogram: Histogram
    feature_projection: FeatureProjection
