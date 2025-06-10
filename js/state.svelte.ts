class Value<T> {
  value: T;

  constructor(value: T) {
    this.value = $state(value);
  }
}

export let rootDiv: Value<HTMLElement>;
export let marginalPlotCompareToBaseProbs: Value<boolean>;
export let cmCompareToWhole: Value<boolean>;
export let wrapTextExampleActivations: Value<boolean>;
export let exampleActivationsIntervalKey: Value<number>;
export let wrapTextFeatureTesting: Value<boolean>;
export let hidePaddingFeatureTesting: Value<boolean>;

export function setupState(element: HTMLElement) {
  rootDiv = new Value(element);
  marginalPlotCompareToBaseProbs = new Value(false);
  cmCompareToWhole = new Value(false);
  wrapTextExampleActivations = new Value(false);
  exampleActivationsIntervalKey = new Value(0);
  wrapTextFeatureTesting = new Value(false);
  hidePaddingFeatureTesting = new Value(false);
}
