// Aligned with types.py

export type Histogram = {
  counts: number[];
  thresholds: number[];
};

export type FeatureTokenSequence = {
  token: string[];
  activation: number[];
  max_index: number;
  extras: Record<string, string[]>;
};

export type CumSumPercentL1Norm = {
  n_neurons: number[];
  cum_sum: number[];
};

export type SequenceInterval = {
  min_activation: number;
  max_activation: number;
  sequences: FeatureTokenSequence[];
};

export type FeatureData = {
  sae_id: number;
  feature_id: number;
  activation_rate: number;
  max_activation: number;
  n_neurons_majority_l1_norm: number;
  cumsum_percent_l1_norm: CumSumPercentL1Norm;
  activations_histogram: Histogram;
  sequence_intervals: Record<string, SequenceInterval>;
};

export type FeatureProjection = {
  feature_id: number[];
  x: number[];
  y: number[];
};

export type CumSumPercentL1NormRange = {
  mins: number[];
  maxs: number[];
};

export type SAEData = {
  sae_id: string;
  num_total_features: number;
  num_alive_features: number;
  num_dead_features: number;
  num_non_activating_features: number;
  alive_feature_ids: number[];
  activation_rate_histogram: Histogram;
  dimensionality_histogram: Histogram;
  cumsum_percent_l1_norm_range: CumSumPercentL1NormRange;
  feature_projection: FeatureProjection;
};

export type Model = {
  height: number;
  sae_ids: string[];
  sae_id: string;
  feature_id: number;
  sae_data: SAEData;
  feature_data: FeatureData;
};

// Front end only

export type Tab = "overview" | "features";

export type FeatureToken = {
  token: string;
  activation: number;
  extras: { key: string; value: string }[];
};
