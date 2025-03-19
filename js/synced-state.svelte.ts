import type { AnyModel } from "@anywidget/types";
import type { FeatureData, DataModel, SAEData, ModelInfo } from "./types.js";

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

export let height: SyncedState<number>;
export let model_info: SyncedState<ModelInfo>;
export let sae_ids: SyncedState<string[]>;
export let sae_id: SyncedState<string>;
export let sae_data: SyncedState<SAEData>;
export let feature_id: SyncedState<number>;
export let feature_data: SyncedState<FeatureData>;

export function setupSyncedState(model: AnyModel<DataModel>) {
  height = createSyncedState("height", model);
  model_info = createSyncedState("model_info", model);
  sae_ids = createSyncedState("sae_ids", model);
  sae_id = createSyncedState("sae_id", model);
  sae_data = createSyncedState("sae_data", model);
  feature_id = createSyncedState("feature_id", model);
  feature_data = createSyncedState("feature_data", model);
}
