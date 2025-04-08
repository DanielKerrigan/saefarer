<script lang="ts">
  import RankingSelect from "./RankingSelect.svelte";
  import { format } from "d3-format";
  import { base_font_size, table_features } from "../synced-state.svelte";
  import Histogram from "./vis/Histogram.svelte";
  import MarginalEffectsPlot from "./vis/MarginalEffectsPlot.svelte";
  import TokenSequences from "./TokenSequences.svelte";
  import { scaleSequential } from "d3-scale";
  import { interpolateBuPu } from "d3-scale-chromatic";

  let {
    onClickFeature,
  }: {
    onClickFeature: (feature_id: number) => void;
  } = $props();

  const activationRateFormat = format(".1~e");

  const dividerWidth = 1;
  const cellPaddingX = $derived(base_font_size.value * 0.5);
  const cellPaddingY = $derived(base_font_size.value * 0.25);
  const contentRowHeight = $derived(base_font_size.value * 3);
  const totalRowHeight = $derived(
    contentRowHeight + dividerWidth + 2 * cellPaddingY,
  );
  const visWidth = $derived(contentRowHeight * 3);
</script>

<div class="sae-container">
  <div class="sae-content">
    <div class="sae-controls">
      <RankingSelect />
    </div>
    <div
      class="sae-table"
      style:--cell-padding-x="{cellPaddingX}px"
      style:--cell-padding-y="{cellPaddingY}px"
    >
      <div class="sae-cell sae-header sae-number-col-header">Index</div>
      <div class="sae-cell sae-header sae-number-col-header">Act. Rate</div>
      <div class="sae-cell sae-header">Act. Distribution</div>
      <div class="sae-cell sae-header">Effect</div>
      <div class="sae-cell sae-header">Example</div>

      {#each table_features.value as feature, i}
        <div
          class="sae-cell sae-number-col-value"
          class:border={i !== table_features.value.length - 1}
        >
          <div>
            <button class="sae-id-btn" onclick={() => onClickFeature(i)}>
              {feature.feature_id}
            </button>
          </div>
        </div>
        <div
          class="sae-cell sae-number-col-value"
          class:border={i !== table_features.value.length - 1}
        >
          <div>
            {activationRateFormat(feature.sequence_act_rate)}
          </div>
        </div>
        <div
          class="sae-cell"
          class:border={i !== table_features.value.length - 1}
        >
          <Histogram
            data={feature.token_acts_histogram}
            width={visWidth}
            height={contentRowHeight}
          />
        </div>
        <div
          class="sae-cell"
          class:border={i !== table_features.value.length - 1}
        >
          <MarginalEffectsPlot
            marginalEffects={feature.marginal_effects}
            width={visWidth}
            height={contentRowHeight}
            showColorLegend={false}
            marginTop={2}
            marginRight={2}
            marginBottom={2}
            marginLeft={2}
            circleRadius={0}
            showXAxis={false}
            showYAxis={false}
          />
        </div>
        <div
          class="sae-cell sae-example-sequence"
          class:border={i !== table_features.value.length - 1}
        >
          <TokenSequences
            color={scaleSequential([0, feature.max_act], interpolateBuPu)}
            sequences={feature.sequence_intervals[
              "Max Activations"
            ].sequences.slice(0, 1)}
            wrap={false}
          />
        </div>
      {/each}
    </div>
  </div>
</div>

<style>
  .sae-container {
    max-height: 100%;
    max-width: 100%;
    height: 100%;
    width: 100%;

    display: flex;
    justify-content: center;
  }

  .sae-content {
    display: flex;
    flex-direction: column;
    gap: 0.5em;
  }

  .sae-table {
    display: grid;
    grid-template-columns:
      max-content
      max-content
      max-content
      max-content
      minmax(0, 1fr);
  }

  .sae-header {
    text-transform: uppercase;
    font-weight: 500;
  }

  .sae-number-col-header {
    text-align: end;
  }

  .sae-cell {
    padding: var(--cell-padding-y) var(--cell-padding-x);
  }

  .sae-number-col-value {
    font-family: var(--font-mono);
    display: flex;
    justify-content: end;
    align-items: center;
  }

  .border {
    border-bottom: 1px solid var(--color-neutral-300);
  }

  .sae-id-btn {
    padding: 0.25em 0.5em;
  }

  .sae-example-sequence {
    display: flex;
    align-items: center;
  }
</style>
