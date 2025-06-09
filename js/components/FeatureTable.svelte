<script lang="ts">
  import RankingControls from "./RankingControls.svelte";
  import { font_sizes, table_features } from "../synced-state.svelte";
  import Histogram from "./vis/Histogram.svelte";
  import MarginalEffectsHeatmap from "./vis/MarginalEffectsHeatmap.svelte";
  import { descending } from "d3-array";
  import { scaleSequential } from "d3-scale";
  import { interpolateBlues, interpolatePlasma } from "d3-scale-chromatic";
  import TokenSequence from "./TokenSequence.svelte";
  import PageControls from "./PageControls.svelte";
  import type { FeatureData } from "../types";
  import {
    activationRatePctFormat,
    actValueHistogramTooltipData,
  } from "./vis/vis-utils";
  import TooltipButton from "./TooltipButton.svelte";
  import InfoIcon from "./icons/InfoIcon.svelte";
  import QuantitativeColorLegend from "./vis/legends/QuantitativeColorLegend.svelte";

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

  const tooltipEnabled = true;
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
      <span>ID</span>
      <TooltipButton position="right">
        {#snippet trigger()}
          <InfoIcon />
        {/snippet}
        {#snippet content()}
          <div class="sae-info">The index of the feature in the SAE.</div>
        {/snippet}
      </TooltipButton>
    </div>
    <div class="sae-table-cell sae-table-header sae-table-header-align-right">
      <span>Act. Rate</span>
      <TooltipButton position="right">
        {#snippet trigger()}
          <InfoIcon />
        {/snippet}
        {#snippet content()}
          <div class="sae-info">
            The percentage of instances that activate the feature.
          </div>
        {/snippet}
      </TooltipButton>
    </div>
    <div class="sae-table-cell sae-table-header sae-table-header-align-right">
      <span>Act. Distribution</span>
      <TooltipButton position="right">
        {#snippet trigger()}
          <InfoIcon />
        {/snippet}
        {#snippet content()}
          <div class="sae-info">
            A histogram of the feature's instance-level activation values.
          </div>
        {/snippet}
      </TooltipButton>
    </div>
    <div class="sae-table-cell sae-table-header sae-table-header-align-right">
      <span>Top Class Probabilities</span>
      <TooltipButton position="right">
        {#snippet trigger()}
          <InfoIcon />
        {/snippet}
        {#snippet content()}
          <div class="sae-info sae-probabilities-info">
            <div>
              The probabilities of the top classes for instances that activate
              the feature. The x-axis encodes the activation value.
            </div>

            <QuantitativeColorLegend
              color={scaleSequential([0, 1], interpolateBlues)}
              width={font_sizes.sm * 16}
              height={56}
              orientation={"horizontal"}
              title="Mean predicted probability"
              marginTop={18}
              marginBottom={24}
              marginLeft={font_sizes.sm * 2}
              marginRight={font_sizes.sm * 2}
              titleFontSize={font_sizes.sm}
              tickLabelFontSize={font_sizes.xs}
            />
          </div>
        {/snippet}
      </TooltipButton>
    </div>
    <div class="sae-table-cell sae-table-header">
      <span>Example</span>
      <TooltipButton position="right">
        {#snippet trigger()}
          <InfoIcon />
        {/snippet}
        {#snippet content()}
          <div class="sae-info sae-example-info">
            <div>
              The token that maximally activates the feature and its surrounding
              context.
            </div>

            <QuantitativeColorLegend
              color={scaleSequential([0, 1], (d) => interpolatePlasma(1 - d))}
              width={font_sizes.sm * 16}
              height={56}
              orientation={"horizontal"}
              title="Activation value"
              marginTop={18}
              marginBottom={24}
              marginLeft={font_sizes.sm * 2}
              marginRight={font_sizes.sm * 2}
              titleFontSize={font_sizes.sm}
              tickLabelFontSize={font_sizes.xs}
              tickValues={[0, 1]}
              tickFormat={(d) => (d === 0 ? "Min" : d === 1 ? "Max" : "")}
            />
          </div>
        {/snippet}
      </TooltipButton>
    </div>

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
          {tooltipEnabled}
          tooltipData={actValueHistogramTooltipData}
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
          {tooltipEnabled}
        />
      </div>
      <div
        class="sae-table-cell sae-table-example-sequence"
        class:sae-table-border={showBorder}
      >
        <TokenSequence
          colorScale={scaleSequential([0, feature.max_act], (d) =>
            interpolatePlasma(1 - d),
          )}
          sequence={feature.sequence_intervals[0].sequences[0]}
          wrap={false}
          {tooltipEnabled}
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
    position: sticky;
    top: 0;
    z-index: 10;
    background-color: var(--color-white);
    display: flex;
    align-items: center;
    gap: 0.25em;
  }

  .sae-table-header > span {
    text-transform: uppercase;
    font-weight: var(--font-medium);
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

  .sae-info {
    font-size: var(--text-sm);
    text-align: start;
    max-width: 16em;
  }

  .sae-example-info,
  .sae-probabilities-info {
    display: flex;
    flex-direction: column;
    gap: 0.5em;
  }
</style>
