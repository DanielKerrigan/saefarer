<script lang="ts">
  import RankingControls from "./RankingControls.svelte";
  import { format } from "d3-format";
  import { base_font_size, table_features } from "../synced-state.svelte";
  import Histogram from "./vis/Histogram.svelte";
  import MarginalEffectsPlot from "./vis/MarginalEffectsPlot.svelte";
  import { scaleSequential } from "d3-scale";
  import type { ScaleOrdinal } from "d3-scale";
  import { interpolateBuPu } from "d3-scale-chromatic";
  import TokenSequence from "./TokenSequence.svelte";
  import PageControls from "./PageControls.svelte";

  let {
    labelColor,
    onClickFeature,
  }: {
    labelColor: ScaleOrdinal<number, string>;
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

<div class="sae-table-container">
  <div class="sae-table-controls">
    <RankingControls />
  </div>
  <div
    class="sae-table"
    style:--cell-padding-x="{cellPaddingX}px"
    style:--cell-padding-y="{cellPaddingY}px"
  >
    <div class="sae-table-cell sae-table-header sae-table-number-header">
      ID
    </div>
    <div class="sae-table-cell sae-table-header sae-table-number-header">
      Act. Rate
    </div>
    <div class="sae-table-cell sae-table-header">Act. Distribution</div>
    <div class="sae-table-cell sae-table-header">Probabilities</div>
    <div class="sae-table-cell sae-table-header">Example</div>

    {#each table_features.value as feature, i}
      {@const showBorder = i !== table_features.value.length - 1}
      <div
        class="sae-table-cell sae-table-number-value"
        class:sae-table-border={showBorder}
      >
        <div>
          <button
            class="sae-table-feature-id-btn"
            onclick={() => onClickFeature(feature.feature_id)}
          >
            {feature.feature_id}
          </button>
        </div>
      </div>
      <div
        class="sae-table-cell sae-table-number-value"
        class:sae-table-border={showBorder}
      >
        <div>
          {activationRateFormat(feature.sequence_act_rate)}
        </div>
      </div>
      <div class="sae-table-cell" class:sae-table-border={showBorder}>
        <Histogram
          data={feature.token_acts_histogram}
          width={visWidth}
          height={contentRowHeight}
        />
      </div>
      <div class="sae-table-cell" class:sae-table-border={showBorder}>
        <MarginalEffectsPlot
          marginalEffects={feature.marginal_effects}
          width={visWidth}
          height={contentRowHeight}
          color={labelColor}
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
        class="sae-table-cell sae-table-example-sequence"
        class:sae-table-border={showBorder}
      >
        <TokenSequence
          color={scaleSequential([0, feature.max_act], interpolateBuPu)}
          sequence={feature.sequence_intervals["Max Activations"].sequences[0]}
          wrap={false}
        />
      </div>
    {/each}
  </div>
  <div class="sae-table-pagination">
    <PageControls />
  </div>
</div>

<style>
  .sae-table-container {
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 0.5em;
  }

  .sae-table {
    min-height: 0;
    overflow-y: auto;
    display: grid;
    grid-template-columns:
      max-content
      max-content
      max-content
      max-content
      minmax(0, 1fr);
    border-top: 1px solid var(--color-neutral-500);
    border-bottom: 1px solid var(--color-neutral-500);
  }

  .sae-table-cell {
    padding: var(--cell-padding-y) var(--cell-padding-x);
  }

  .sae-table-header {
    text-transform: uppercase;
    font-weight: 500;
    position: sticky;
    top: 0;
    z-index: 10;
    background-color: var(--color-white);
  }

  .sae-table-number-header {
    text-align: end;
  }

  .sae-table-number-value {
    font-family: var(--font-mono);
    display: flex;
    justify-content: end;
    align-items: center;
  }

  .sae-table-border {
    border-bottom: 1px solid var(--color-neutral-300);
  }

  .sae-table-feature-id-btn {
    padding: 0.25em 0.5em;
  }

  .sae-table-example-sequence {
    display: flex;
    align-items: center;
    overflow-x: auto;
  }
</style>
