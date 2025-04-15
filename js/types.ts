// Aligned with types.py

export type HistogramData = {
  counts: number[];
  thresholds: number[];
};

export type MarginalEffectsData = {
  probs: number[][];
  thresholds: number[];
  non_act_probs: number[];
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
  pred_label: number;
  pred_probs: number[];
};

export type SequenceInterval = {
  min_max_act: number;
  max_max_act: number;
  sequences: FeatureTokenSequence[];
};

export type ConfusionMatrixCell = {
  label: number;
  pred_label: number;
  count: number;
  pct: number;
};

export type ConfusionMatrixData = {
  n_sequences: number;
  error_count: number;
  error_pct: number;
  cells: ConfusionMatrixCell[];
  label_counts: number[];
  label_pcts: number[];
  pred_label_counts: number[];
  pred_label_pcts: number[];
  false_pos_counts: number[];
  false_pos_pcts: number[];
  false_neg_counts: number[];
  false_neg_pcts: number[];
};

export type FeatureData = {
  sae_id: number;
  feature_id: number;
  max_act: number;
  token_act_rate: number;
  token_acts_histogram: HistogramData;
  sequence_act_rate: number;
  sequence_acts_histogram: HistogramData;
  marginal_effects: MarginalEffectsData;
  cm: ConfusionMatrixData;
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
  n_total_features: number;
  n_alive_features: number;
  n_dead_features: number;
  n_non_activating_features: number;
  alive_feature_ids: number[];
  token_act_rate_histogram: HistogramData;
  sequence_act_rate_histogram: HistogramData;
  feature_projection: FeatureProjection;
};

export type FeatureIdRankingOption = {
  kind: "feature_id";
  descending: boolean;
};

export type SequenceActRateRankingOption = {
  kind: "sequence_act_rate";
  descending: boolean;
};

export type LabelRankingOption = {
  kind: "label";
  true_label: string;
  pred_label: string;
  descending: boolean;
};

export type RankingOption =
  | FeatureIdRankingOption
  | SequenceActRateRankingOption
  | LabelRankingOption;

export type ModelInfo = {
  labels: string[];
  label_indices: number[];
  cm: ConfusionMatrixData;
  mean_pred_label_probs: number[];
};

export type DataModel = {
  height: number;
  base_font_size: number;
  n_table_rows: number;
  model_info: ModelInfo;
  sae_ids: string[];
  sae_id: string;
  sae_data: SAEData;
  table_ranking_option: RankingOption;
  table_min_act_rate: number;
  table_page_index: number;
  max_table_page_index: number;
  table_features: FeatureData[];
  detail_feature: FeatureData;
  detail_feature_id: number;
};

// JS only

export type Tab = "overview" | "table" | "detail";
