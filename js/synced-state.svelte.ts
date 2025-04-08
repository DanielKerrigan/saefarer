import type { AnyModel } from "@anywidget/types";
import type { DataModel } from "./types.js";

type SyncedState<T> = { value: T };

export function createSyncedState<K extends keyof DataModel>(
  key: K,
  model: AnyModel<DataModel>,
): SyncedState<DataModel[K]> {
  let value = $state(model.get(key));

  model.on(`change:${key}`, () => (value = model.get(key)));

  return {
    get value() {
      return value;
    },
    set value(v: DataModel[K]) {
      model.set(key, v);
      model.save_changes();
    },
  };
}

class TwoWaySyncedState<K extends keyof DataModel> {
  #key: K;
  #model: AnyModel<DataModel>;
  // TODO: Update this when this issue is resolved
  // https://github.com/sveltejs/svelte/issues/12655
  #value: DataModel[K] = $state()!;

  constructor(key: K, model: AnyModel<DataModel>) {
    this.#key = key;
    this.#model = model;
    this.#value = this.#model.get(this.#key);
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
  // TODO: Update this when this issue is resolved
  // https://github.com/sveltejs/svelte/issues/12655
  #value: DataModel[K] = $state()!;

  constructor(key: K, model: AnyModel<DataModel>) {
    this.#value = model.get(key);
    model.on(`change:${key}`, () => (this.#value = model.get(key)));
  }

  get value() {
    return this.#value;
  }
}

export let height: OneWaySyncedState<"height">;
export let base_font_size: OneWaySyncedState<"base_font_size">;
export let n_table_rows: OneWaySyncedState<"n_table_rows">;
export let model_info: OneWaySyncedState<"model_info">;
export let sae_ids: OneWaySyncedState<"sae_ids">;
export let sae_id: OneWaySyncedState<"sae_id">;
export let sae_data: OneWaySyncedState<"sae_data">;
export let table_ranking_option: TwoWaySyncedState<"table_ranking_option">;
export let table_page_index: TwoWaySyncedState<"table_page_index">;
export let max_table_page_index: OneWaySyncedState<"max_table_page_index">;
export let table_features: OneWaySyncedState<"table_features">;
export let detail_feature: OneWaySyncedState<"detail_feature">;
export let detail_feature_id: TwoWaySyncedState<"detail_feature_id">;

export function setupSyncedState(model: AnyModel<DataModel>) {
  height = new OneWaySyncedState("height", model);
  base_font_size = new OneWaySyncedState("base_font_size", model);
  n_table_rows = new OneWaySyncedState("n_table_rows", model);
  model_info = new OneWaySyncedState("model_info", model);
  sae_ids = new OneWaySyncedState("sae_ids", model);
  sae_id = new OneWaySyncedState("sae_id", model);
  sae_data = new OneWaySyncedState("sae_data", model);
  table_ranking_option = new TwoWaySyncedState("table_ranking_option", model);
  table_page_index = new TwoWaySyncedState("table_page_index", model);
  max_table_page_index = new OneWaySyncedState("max_table_page_index", model);
  table_features = new OneWaySyncedState("table_features", model);
  detail_feature = new OneWaySyncedState("detail_feature", model);
  detail_feature_id = new TwoWaySyncedState("detail_feature_id", model);
}
