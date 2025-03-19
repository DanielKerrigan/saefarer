// Aligned with types.py

export type Histogram = {
  counts: number[];
  thresholds: number[];
};

export type MarginalEffects = {
  probs: number[][];
  thresholds: number[];
};

export type DisplayToken = {
  display: string;
  token_ids: number[];
  acts: number[];
  max_act: number;
  extras: Record<string, string[]>;
  is_special: boolean;
};

export type FeatureTokenSequence = {
  sequence_index: number;
  display_tokens: DisplayToken[];
  max_token_index: number;
  label: number;
  predicted_label: number;
  pred_probs: number[];
};

export type SequenceInterval = {
  min_max_act: number;
  max_max_act: number;
  sequences: FeatureTokenSequence[];
};

export type FeatureData = {
  sae_id: number;
  feature_id: number;
  max_act: number;
  token_act_rate: number;
  token_acts_histogram: Histogram;
  sequence_act_rate: number;
  sequence_acts_histogram: Histogram;
  marginal_effects: MarginalEffects;
  sequence_intervals: Record<string, SequenceInterval>;
  mean_pred_label_probs: number[];
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
  token_act_rate_histogram: Histogram;
  sequence_act_rate_histogram: Histogram;
  feature_projection: FeatureProjection;
};

export type ConfusionMatrixCell = {
  label: number;
  pred_label: number;
  count: number;
};

export type ModelInfo = {
  n_sequences: number;
  labels: string[];
  label_indices: number[];
  cm: ConfusionMatrixCell[];
  mean_pred_label_probs: number[];
  label_counts: number[];
  pred_label_counts: number[];
};

export type DataModel = {
  height: number;
  model_info: ModelInfo;
  sae_ids: string[];
  sae_id: string;
  sae_data: SAEData;
  feature_id: number;
  feature_data: FeatureData;
};

// JS only

export type Tab = "overview" | "features";

export type FeatureToken = {
  token: string;
  activation: number;
  extras: { key: string; value: string }[];
};
