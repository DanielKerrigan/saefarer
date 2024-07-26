export type Histogram = {
  counts: number[];
  thresholds: number[];
};

export type TokenSequence = {
  token: string[];
  activation: number[];
  max_index: number;
};

export type FeatureData = {
  firing_rate: number;
  activations_histogram: Histogram;
  sequences: TokenSequence[];
};

export type SAEData = {
  num_dead_features: number;
  num_alive_features: number;
  firing_rate_histogram: Histogram;
  dead_feature_indices: number[];
  alive_feature_indices: number[];
  feature_index_to_path: Record<number, string>;
};

export type Model = {
  height: number;
  sae_data: SAEData;
  feature_index: number;
  feature_data: FeatureData;
};

export type Tab = "overview" | "features";
