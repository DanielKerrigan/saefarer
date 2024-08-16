export type Histogram = {
  counts: number[];
  thresholds: number[];
};

export type TokenSequence = {
  token: string[];
  activation: number[];
  max_index: number;
};

export type CumSumPercentL1Norm = {
  n_neurons: number[];
  cum_sum: number[];
};

export type FeatureData = {
  sae_id: number;
  feature_id: number;
  activation_rate: number;
  max_activation: number;
  n_neurons_majority_l1_norm: number;
  cumsum_percent_l1_norm: CumSumPercentL1Norm;
  activations_histogram: Histogram;
  sequences: Record<string, TokenSequence[]>;
};

export type FeatureProjection = {
  feature_id: number[];
  x: number[];
  y: number[];
};

export type CumSumPercentL1NormRange = {
  min: number[];
  max: number[];
};

export type SAEData = {
  sae_id: string;
  num_alive_features: number;
  num_dead_features: number;
  alive_feature_ids: number[];
  dead_feature_ids: number[];
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

export type Tab = "overview" | "features";
