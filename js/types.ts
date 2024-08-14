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
  feature_id: number;
  sae_id: string;
  firing_rate: number;
  activations_histogram: Histogram;
  sequences: TokenSequence[];
};

export type FeatureProjection = {
  feature_id: number[];
  x: number[];
  y: number[];
};

export type SAEData = {
  sae_id: string;
  num_alive_features: number;
  num_dead_features: number;
  alive_feature_ids: number[];
  dead_feature_ids: number[];
  firing_rate_histogram: Histogram;
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
