// Aligned with types.py

export type Histogram = {
  counts: number[];
  thresholds: number[];
};

export type MarginalEffects = {
  probabilities: number[][];
  thresholds: number[];
};

export type DisplayToken = {
  display: string;
  token_ids: number[];
  activations: number[];
  max_activation: number;
  extras: Record<string, string[]>;
};

export type FeatureTokenSequence = {
  display_tokens: DisplayToken[];
  max_index: number;
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
  activations_histogram: Histogram;
  marginal_effects: MarginalEffects;
  sequence_intervals: Record<string, SequenceInterval>;
};

export type FeatureProjection = {
  feature_ids: number[];
  xs: number[];
  ys: number[];
};

export type SAEData = {
  sae_id: string;
  num_total_features: number;
  num_alive_features: number;
  num_dead_features: number;
  num_non_activating_features: number;
  alive_feature_ids: number[];
  activation_rate_histogram: Histogram;
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
