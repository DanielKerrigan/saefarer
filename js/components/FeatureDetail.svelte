<script lang="ts">
  import { scaleSequential } from "d3-scale";
  import { interpolatePlasma } from "d3-scale-chromatic";
  import {
    can_inference,
    dataset_info,
    detail_feature,
    detail_feature_id,
    model_info,
    sae_data,
  } from "../synced-state.svelte";
  import FeatureTokenSequenceTable from "./FeatureTokenSequenceTable.svelte";
  import ConfusionMatrix from "./vis/ConfusionMatrix.svelte";
  import {
    activationRatePctFormat,
    countFormat,
    getSizeWithAspectRatio,
    getSizeWithAspectRatioMargins,
  } from "./vis/vis-utils";
  import MarginalEffectsHeatmap from "./vis/MarginalEffectsHeatmap.svelte";
  import FeatureTesting from "./FeatureTesting.svelte";

  let {}: {} = $props();

  const maxNumDigits = $derived(
    Math.log10(sae_data.value.n_total_features) + 1,
  );

  const tokenColor = $derived(
    scaleSequential()
      .domain([0, detail_feature.value.max_act])
      .interpolator((d) => interpolatePlasma(1 - d)),
  );

  let featureIdInputValue = $derived(detail_feature_id.value);

  function onClickGo() {
    detail_feature_id.value = featureIdInputValue;
  }

  let maxEffectHeight = $state(0);
  let maxEffectWidth = $state(0);
  let effectSize = $derived(
    getSizeWithAspectRatio(maxEffectWidth, maxEffectHeight, 1.6),
  );

  const cmMarginTop = 8;
  const cmMarginRight = 88;
  const cmMarginBottom = 80;
  const cmMarginLeft = 80;

  let maxCMWidth = $state(0);
  let maxCMHeight = $state(0);

  const cmSize = $derived(
    getSizeWithAspectRatioMargins(
      maxCMWidth,
      maxCMHeight,
      1,
      cmMarginTop,
      cmMarginRight,
      cmMarginBottom,
      cmMarginLeft,
    ),
  );

  let marginalCompareToBase = $state(false);
  let cmCompareToWhole = $state(false);
</script>

<div class="sae-container">
  <div class="sae-controls">
    <div class="sae-feature-input">
      <label>
        <span style:font-weight="var(--font-medium)">Feature ID:</span>
        <input
          type="number"
          style:width="{maxNumDigits + 1}em"
          bind:value={featureIdInputValue}
        />
      </label>
      <button onclick={onClickGo}>Go</button>
    </div>
    <div>
      <span style:font-weight="var(--font-medium)">Activation Rate:</span>
      <span>
        {activationRatePctFormat(detail_feature.value.sequence_act_rate)} ({countFormat(
          detail_feature.value.cm.n_sequences,
        )} instances)
      </span>
    </div>
  </div>
  <div
    class={[
      "sae-main",
      can_inference.value ? "sae-grid-inference" : "sae-grid-no-inference",
    ]}
  >
    <div class="sae-effects-container">
      <div class="sae-effects-controls">
        <div style:font-weight="var(--font-medium)">
          Predicted Probabilities
        </div>
        <label>
          <input type="checkbox" bind:checked={marginalCompareToBase} />
          <span>Compare to base probabilities</span>
        </label>
      </div>
      <div
        class="sae-effects-vis"
        bind:clientWidth={maxEffectWidth}
        bind:clientHeight={maxEffectHeight}
      >
        <MarginalEffectsHeatmap
          marginalEffects={detail_feature.value.marginal_effects}
          distribution={detail_feature.value.sequence_acts_histogram}
          classes={dataset_info.value.label_indices}
          compareToBaseProbs={marginalCompareToBase}
          marginTop={32}
          marginRight={88}
          marginLeft={80}
          marginBottom={40}
          width={effectSize.width}
          height={effectSize.height}
          xAxisLabel={"Activation value"}
          yAxisLabel={"Predicted label"}
          showColorLegend={true}
        />
      </div>
    </div>

    <div class="sae-cm-container">
      <div class="sae-cm-controls">
        <div style:font-weight="var(--font-medium)">Confusion Matrix</div>
        <label>
          <input type="checkbox" bind:checked={cmCompareToWhole} />
          <span>Compare to whole dataset</span>
        </label>
      </div>
      <div
        class="sae-cm-vis"
        bind:clientWidth={maxCMWidth}
        bind:clientHeight={maxCMHeight}
      >
        <ConfusionMatrix
          cm={detail_feature.value.cm}
          other={model_info.value.cm}
          showDifference={cmCompareToWhole}
          legend={"vertical"}
          width={cmSize.width}
          height={cmSize.height}
          marginTop={cmMarginTop}
          marginRight={cmMarginRight}
          marginBottom={cmMarginBottom}
          marginLeft={cmMarginLeft}
        />
      </div>
    </div>

    <div class="sae-sequences-container">
      <FeatureTokenSequenceTable {tokenColor} />
    </div>

    {#if can_inference.value}
      <div class="sae-inference-container">
        <FeatureTesting {tokenColor} featureId={detail_feature_id.value} />
      </div>
    {/if}
  </div>
</div>

<style>
  /* overall */

  .sae-container {
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 1.5em;
  }

  label {
    display: flex;
    align-items: center;
    gap: 0.25em;
  }

  /* top menu */

  .sae-controls {
    display: flex;
    gap: 1em;
  }

  .sae-feature-input {
    display: flex;
    align-items: center;
    gap: 0.25em;
  }

  .sae-feature-input label input {
    align-self: flex-start;
    border: 1px solid var(--color-black);
    border-radius: 0.25em;
    padding: 0em 0.25em;
  }

  .sae-feature-input button {
    padding: 0.25em;
  }

  /* main content */

  .sae-main {
    flex: 1;
    min-height: 0;

    display: grid;
    gap: 1.5em;
  }

  .sae-grid-inference {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    grid-template-rows: repeat(12, minmax(0, 1fr));
    grid-template-areas:
      "effects sequences"
      "effects sequences"
      "effects sequences"
      "effects sequences"
      "effects sequences"
      "effects sequences"
      "cm inference"
      "cm inference"
      "cm inference"
      "cm inference"
      "cm inference"
      "cm inference";
  }

  .sae-grid-no-inference {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    grid-template-rows: repeat(2, minmax(0, 1fr));
    grid-template-areas:
      "effects sequences"
      "cm sequences";
  }

  /* marginal plot */

  .sae-effects-container {
    grid-area: effects;
    display: flex;
    flex-direction: column;
  }

  .sae-effects-controls {
    display: flex;
    gap: 1em;
    align-items: center;
    justify-content: flex-start;
  }

  .sae-effects-vis {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 0;
    min-width: 0;
  }

  /* confusion matrix */

  .sae-cm-container {
    grid-area: cm;
    display: flex;
    flex-direction: column;
  }

  .sae-cm-controls {
    display: flex;
    gap: 1em;
    align-items: center;
    justify-content: flex-start;
  }

  .sae-cm-vis {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 0;
    min-width: 0;
  }

  /* sequences */

  .sae-sequences-container {
    grid-area: sequences;
    min-height: 0;
    min-width: 0;
  }

  /* inferencing */

  .sae-inference-container {
    grid-area: inference;
    min-height: 0;
    min-width: 0;
  }
</style>
