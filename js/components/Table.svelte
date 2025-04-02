<script lang="ts">
  import { format } from "d3-format";
  import { base_font_size, features } from "../synced-state.svelte";
  import Histogram from "./vis/Histogram.svelte";
  import MarginalEffectsPlot from "./vis/MarginalEffectsPlot.svelte";
  import Sequence from "./Sequence.svelte";

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

    {#each features.value as feature, i}
      <div
        class="sae-cell sae-number-col-value"
        class:border={i !== features.value.length - 1}
      >
        <div>
          {feature.feature_id}
        </div>
      </div>
      <div
        class="sae-cell sae-number-col-value"
        class:border={i !== features.value.length - 1}
      >
        <div>
          {activationRateFormat(feature.sequence_act_rate)}
        </div>
      </div>
      <div class="sae-cell" class:border={i !== features.value.length - 1}>
        <Histogram
          data={feature.token_acts_histogram}
          width={visWidth}
          height={contentRowHeight}
        />
      </div>
      <div class="sae-cell" class:border={i !== features.value.length - 1}>
        <MarginalEffectsPlot
          data={feature.marginal_effects}
          width={visWidth}
          height={contentRowHeight}
          showColorLegend={false}
        />
      </div>
      <div class="sae-cell" class:border={i !== features.value.length - 1}>
        <Sequence {feature} />
      </div>
    {/each}
  </div>
</div>

<style>
  .sae-table-container {
    max-height: 100%;
    max-width: 100%;
    height: 100%;
    width: 100%;
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
    border-bottom: 1px solid var(--color-neutral-200);
  }
</style>
