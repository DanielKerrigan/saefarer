<script lang="ts">
  import RankingControls from "./RankingControls.svelte";
  import { font_sizes, table_features } from "../synced-state.svelte";
  import Histogram from "./vis/Histogram.svelte";
  import MarginalEffectsHeatmap from "./vis/MarginalEffectsHeatmap.svelte";
  import { descending } from "d3-array";
  import { scaleSequential } from "d3-scale";
  import { interpolateBlues } from "d3-scale-chromatic";
  import TokenSequence from "./TokenSequence.svelte";
  import PageControls from "./PageControls.svelte";
  import type { FeatureData } from "../types";
  import { activationRatePctFormat } from "./vis/vis-utils";

  let {
    onClickFeature,
  }: {
    onClickFeature: (feature_id: number) => void;
  } = $props();

  const dividerWidth = 1;
  const cellPaddingX = $derived(font_sizes.base * 0.5);
  const cellPaddingY = $derived(font_sizes.base * 0.25);
  const contentRowHeight = $derived(font_sizes.base * 3);
  const totalRowHeight = $derived(
    contentRowHeight + dividerWidth + 2 * cellPaddingY,
  );
  const visWidth = $derived(contentRowHeight * 3);
  const marginalPlotMarginLeft = 80;

  function getTopClasses(feature: FeatureData): number[] {
    return feature.cm.pred_label_pcts
      .map((pct, label) => ({ pct, label }))
      .sort((a, b) => descending(a.pct, b.pct))
      .slice(0, 3)
      .map(({ label }) => label);
  }
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
    <div class="sae-table-cell sae-table-header sae-table-header-align-right">
      ID
    </div>
    <div class="sae-table-cell sae-table-header sae-table-header-align-right">
      Act. Rate
    </div>
    <div class="sae-table-cell sae-table-header sae-table-header-align-right">
      Act. Distribution
    </div>
    <div class="sae-table-cell sae-table-header sae-table-header-align-right">
      Top Class Probabilities
    </div>
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
          {activationRatePctFormat(feature.sequence_act_rate)}
        </div>
      </div>
      <div class="sae-table-cell" class:sae-table-border={showBorder}>
        <Histogram
          data={feature.sequence_acts_histogram}
          width={visWidth}
          height={contentRowHeight}
          tooltipEnabled={false}
        />
      </div>
      <div class="sae-table-cell" class:sae-table-border={showBorder}>
        <MarginalEffectsHeatmap
          marginalEffects={feature.marginal_effects}
          classes={getTopClasses(feature)}
          width={visWidth + marginalPlotMarginLeft}
          height={contentRowHeight}
          maxColorDomain={1}
          showColorLegend={false}
          marginTop={0}
          marginRight={0}
          marginBottom={0}
          marginLeft={marginalPlotMarginLeft}
          showXAxis={false}
          showYAxis={true}
          tooltipEnabled={false}
        />
      </div>
      <div
        class="sae-table-cell sae-table-example-sequence"
        class:sae-table-border={showBorder}
      >
        <TokenSequence
          colorScale={scaleSequential([0, feature.max_act], interpolateBlues)}
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
    font-weight: var(--font-medium);
    position: sticky;
    top: 0;
    z-index: 10;
    background-color: var(--color-white);
  }

  .sae-table-header-align-right {
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
