import type { AnyModel } from "@anywidget/types";
import type { DataModel } from "./types.js";

class TwoWaySyncedState<K extends keyof DataModel> {
  #key: K;
  #model: AnyModel<DataModel>;
  #value: DataModel[K];

  constructor(key: K, model: AnyModel<DataModel>) {
    this.#key = key;
    this.#model = model;
    this.#value = $state(this.#model.get(this.#key));
    this.#model.on(
      `change:${this.#key}`,
      () => (this.#value = this.#model.get(this.#key)),
    );
  }

  get value() {
    return this.#value;
  }

  set value(v: DataModel[K]) {
    this.#model.set(this.#key, v);
    this.#model.save_changes();
  }
}

class OneWaySyncedState<K extends keyof DataModel> {
  #value: DataModel[K];

  constructor(key: K, model: AnyModel<DataModel>) {
    this.#value = $state(model.get(key));
    model.on(`change:${key}`, () => (this.#value = model.get(key)));
  }

  get value() {
    return this.#value;
  }
}

class FontSizes {
  #base: OneWaySyncedState<"base_font_size">;
  #xs: number;
  #sm: number;
  #lg: number;
  #xl: number;

  constructor(model: AnyModel<DataModel>) {
    this.#base = new OneWaySyncedState("base_font_size", model);
    this.#xs = $derived(this.#base.value * 0.75);
    this.#sm = $derived(this.#base.value * 0.875);
    this.#lg = $derived(this.#base.value * 1.125);
    this.#xl = $derived(this.#base.value * 1.25);
  }

  get base() {
    return this.#base.value;
  }

  get xs() {
    return this.#xs;
  }

  get sm() {
    return this.#sm;
  }

  get lg() {
    return this.#lg;
  }

  get xl() {
    return this.#xl;
  }
}

// one-way synced state
export let height: OneWaySyncedState<"height">;
export let n_table_rows: OneWaySyncedState<"n_table_rows">;
export let model_info: OneWaySyncedState<"model_info">;
export let sae_ids: OneWaySyncedState<"sae_ids">;
export let sae_id: OneWaySyncedState<"sae_id">;
export let sae_data: OneWaySyncedState<"sae_data">;

// two-way synced state
export let table_ranking_option: TwoWaySyncedState<"table_ranking_option">;
export let table_min_act_rate: TwoWaySyncedState<"table_min_act_rate">;
export let table_page_index: TwoWaySyncedState<"table_page_index">;
export let max_table_page_index: OneWaySyncedState<"max_table_page_index">;
export let table_features: OneWaySyncedState<"table_features">;
export let detail_feature: OneWaySyncedState<"detail_feature">;
export let detail_feature_id: TwoWaySyncedState<"detail_feature_id">;

// derived state
export let font_sizes: FontSizes;

export function setupSyncedState(model: AnyModel<DataModel>) {
  // one-way synced state
  height = new OneWaySyncedState("height", model);
  n_table_rows = new OneWaySyncedState("n_table_rows", model);
  model_info = new OneWaySyncedState("model_info", model);
  sae_ids = new OneWaySyncedState("sae_ids", model);
  sae_id = new OneWaySyncedState("sae_id", model);
  sae_data = new OneWaySyncedState("sae_data", model);

  // two-way synced state
  table_ranking_option = new TwoWaySyncedState("table_ranking_option", model);
  table_min_act_rate = new TwoWaySyncedState("table_min_act_rate", model);
  table_page_index = new TwoWaySyncedState("table_page_index", model);
  max_table_page_index = new OneWaySyncedState("max_table_page_index", model);
  table_features = new OneWaySyncedState("table_features", model);
  detail_feature = new OneWaySyncedState("detail_feature", model);
  detail_feature_id = new TwoWaySyncedState("detail_feature_id", model);

  // derived state
  font_sizes = new FontSizes(model);
}
