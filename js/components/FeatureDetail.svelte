<script lang="ts">
  import { scaleSequential } from "d3-scale";
  import type { ScaleOrdinal } from "d3-scale";
  import { interpolateBuPu } from "d3-scale-chromatic";
  import { format } from "d3-format";
  import {
    base_font_size,
    detail_feature,
    detail_feature_id,
    model_info,
    sae_data,
  } from "../synced-state.svelte";
  import FeatureTokenSequences from "./FeatureTokenSequences.svelte";
  import MarginalEffectsPlot from "./vis/MarginalEffectsPlot.svelte";
  import ConfusionMatrix from "./vis/ConfusionMatrix.svelte";
  import CategoricalColorLegend from "./vis/legends/CategoricalColorLegend.svelte";
  import {
    getSizeWithAspectRatio,
    getSizeWithAspectRatioMargins,
  } from "./vis/vis-utils";

  let {
    labelColor,
  }: {
    labelColor: ScaleOrdinal<number, string>;
  } = $props();

  const percentFormat = format(".3%");

  const maxNumDigits = $derived(
    Math.log10(sae_data.value.n_total_features) + 1,
  );

  const tokenColor = $derived(
    scaleSequential()
      .domain([0, detail_feature.value.max_act])
      .interpolator(interpolateBuPu),
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

  const cmMarginTop = 72;
  const cmMarginRight = 72;
  const cmMarginBottom = 10;
  const cmMarginLeft = 72;

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
</script>

<div class="sae-container">
  <div class="sae-controls">
    <div class="sae-feature-input">
      <label>
        <span>ID</span>
        <input
          type="number"
          style:width="{maxNumDigits + 1}em"
          bind:value={featureIdInputValue}
        />
      </label>
      <button onclick={onClickGo}>Go</button>
    </div>
    <div>
      <span style:font-weight="500">Act. Rate:</span>
      <span>
        {percentFormat(detail_feature.value.sequence_act_rate)} of instances
      </span>
    </div>
    <CategoricalColorLegend
      color={labelColor}
      labels={model_info.value.labels}
      fontSize={base_font_size.value}
      titleFontWeight={500}
      title="Label"
    />
  </div>
  <div class="sae-main">
    <div
      class="sae-effects-container"
      bind:clientWidth={maxEffectWidth}
      bind:clientHeight={maxEffectHeight}
    >
      <MarginalEffectsPlot
        marginalEffects={detail_feature.value.marginal_effects}
        color={labelColor}
        distribution={detail_feature.value.sequence_acts_histogram}
        marginTop={32}
        marginRight={20}
        marginLeft={64}
        marginBottom={40}
        width={effectSize.width}
        height={effectSize.height}
        xAxisLabel={"Activation value →"}
        yAxisLabel={"Mean predicted probability →"}
        showColorLegend={false}
        showBaseValues={true}
      />
    </div>

    <div
      class="sae-cm-container"
      bind:clientWidth={maxCMWidth}
      bind:clientHeight={maxCMHeight}
    >
      <ConfusionMatrix
        cm={detail_feature.value.cm}
        legend={"vertical"}
        width={cmSize.width}
        height={cmSize.height}
        marginTop={cmMarginTop}
        marginRight={cmMarginRight}
        marginBottom={cmMarginBottom}
        marginLeft={cmMarginLeft}
      />
    </div>

    <div class="sae-sequences-container">
      <FeatureTokenSequences {labelColor} {tokenColor} />
    </div>
  </div>
</div>

<style>
  .sae-container {
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 1em;
  }

  .sae-controls {
    display: flex;
    gap: 1em;
  }

  .sae-feature-input {
    display: flex;
    align-items: center;
    gap: 0.25em;
  }

  .sae-feature-input label {
    display: flex;
    align-items: center;
    gap: 0.5em;
  }

  .sae-feature-input label span {
    font-weight: 500;
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

  .sae-main {
    flex: 1;
    min-height: 0;

    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    grid-template-rows: repeat(2, minmax(0, 1fr));
    grid-template-areas:
      "effects sequences"
      "cm sequences";
    gap: 1em;
  }

  .sae-effects-container {
    grid-area: effects;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 0;
    min-width: 0;
  }

  .sae-cm-container {
    grid-area: cm;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 0;
    min-width: 0;
  }

  .sae-sequences-container {
    grid-area: sequences;
    min-height: 0;
    min-width: 0;
  }
</style>
