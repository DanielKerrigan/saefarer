<script lang="ts">
  import {
    dataset_info,
    detail_feature,
    font_sizes,
  } from "../synced-state.svelte";
  import type { ScaleSequential } from "d3-scale";
  import TokenSequence from "./TokenSequence.svelte";
  import QuantitativeColorLegend from "./vis/legends/QuantitativeColorLegend.svelte";
  import TooltipButton from "./TooltipButton.svelte";
  import { activationValueFormat } from "./vis/vis-utils";
  import TooltipTable from "./TooltipTable.svelte";
  import { range } from "d3-array";
  import InfoIcon from "./icons/InfoIcon.svelte";
  import type { FeatureTokenSequence } from "../types";
  import {
    exampleActivationsIntervalKey,
    wrapTextExampleActivations,
  } from "../state.svelte";
  import HelpIcon from "./icons/HelpIcon.svelte";

  let {
    tokenColor,
  }: {
    tokenColor: ScaleSequential<string>;
  } = $props();

  let seqInterval = $derived(
    detail_feature.value.sequence_intervals[
      exampleActivationsIntervalKey.value
    ],
  );

  function getTooltipTableData(seq: FeatureTokenSequence) {
    const extras = Object.entries(seq.extras).map(([key, value]) => ({
      key,
      value,
    }));

    return [
      { key: "Instance index", value: `${seq.sequence_index}` },
      ...extras,
    ];
  }
</script>

<div class="sae-sequence-container">
  <div class="sae-sequences-header">
    <div class="sae-sequences-controls">
      <div class="sae-info">
        <span>Example Activations</span>

        <TooltipButton position="right">
          {#snippet trigger()}
            <HelpIcon />
          {/snippet}
          {#snippet content()}
            <div class="sae-info">
              This section shows snippets of instances that activate the
              feature.
            </div>
          {/snippet}
        </TooltipButton>
      </div>
      <label>
        <span>Range:</span>
        <select bind:value={exampleActivationsIntervalKey.value}>
          <option value={0}> Max activations </option>
          {#each range(detail_feature.value.sequence_intervals.length - 1, 0, -1) as i}
            {@const interval = detail_feature.value.sequence_intervals[i]}
            <option value={i}>
              {activationValueFormat(interval.min_max_act)} to {activationValueFormat(
                interval.max_max_act,
              )}
            </option>
          {/each}
        </select>
      </label>
      <label>
        <input
          type="checkbox"
          bind:checked={wrapTextExampleActivations.value}
        />
        <span>Wrap text</span>
      </label>
    </div>

    <div class="sae-sequences-color-legend">
      <QuantitativeColorLegend
        width={256}
        height={56}
        color={tokenColor}
        orientation="horizontal"
        title="Activation value"
        marginTop={18}
        marginBottom={24}
        titleFontSize={font_sizes.sm}
        tickLabelFontSize={font_sizes.xs}
        tickFormat={(d) => (d === 0 ? "> 0" : activationValueFormat(d))}
      />
    </div>
  </div>

  <div class="sae-sequences-table">
    <div class="sae-sequences-table-cell sae-sequences-table-header"></div>
    <div class="sae-sequences-table-cell sae-sequences-table-header">Pred.</div>
    <div class="sae-sequences-table-cell sae-sequences-table-header">True</div>
    <div class="sae-sequences-table-cell sae-sequences-table-header">
      Tokens
    </div>

    {#each seqInterval.sequences as seq, i}
      {@const showBorder = i !== seqInterval.sequences.length - 1}
      <div
        class="sae-sequences-table-cell"
        class:sae-sequences-table-border={showBorder}
      >
        <TooltipButton position="left">
          {#snippet trigger()}
            <InfoIcon />
          {/snippet}

          {#snippet content()}
            <TooltipTable data={getTooltipTableData(seq)} />
          {/snippet}
        </TooltipButton>
      </div>
      <div
        class="sae-sequences-table-cell"
        class:sae-sequences-table-border={showBorder}
      >
        {dataset_info.value.labels[seq.pred_label]}
      </div>
      <div
        class="sae-sequences-table-cell"
        class:sae-sequences-table-border={showBorder}
      >
        {dataset_info.value.labels[seq.label]}
      </div>
      <div
        class="sae-sequences-table-cell sae-sequences-table-tokens"
        class:sae-sequences-table-border={showBorder}
      >
        <TokenSequence
          colorScale={tokenColor}
          sequence={seq}
          wrap={wrapTextExampleActivations.value}
          hidePadding={false}
        />
      </div>
    {/each}
  </div>
</div>

<style>
  select {
    border: 1px solid var(--color-black);
  }

  label {
    display: flex;
    align-items: center;
    gap: 0.25em;
  }

  .sae-sequence-container {
    min-height: 0;
    max-height: 100%;
    display: flex;
    flex-direction: column;
    gap: 0.5em;
  }

  .sae-sequences-header {
    display: flex;
    flex-direction: column;
    gap: 0.25em;
  }

  .sae-sequences-controls {
    display: flex;
    gap: 1em;
    align-items: center;
    justify-content: flex-start;
  }

  .sae-sequences-table {
    min-height: 0;
    overflow-y: auto;
    display: grid;
    grid-auto-rows: max-content;
    grid-template-columns:
      max-content
      max-content
      max-content
      minmax(0, 1fr);
    border-top: 1px solid var(--color-neutral-500);
    border-bottom: 1px solid var(--color-neutral-500);
  }

  .sae-sequences-table-cell {
    padding: 0.25em 0.5em;
    display: flex;
    align-items: center;
  }

  .sae-sequences-table-header {
    text-transform: uppercase;
    font-weight: var(--font-medium);
    position: sticky;
    top: 0;
    z-index: 10;
    background-color: var(--color-white);
  }

  .sae-sequences-table-border {
    border-bottom: 1px solid var(--color-neutral-300);
  }

  .sae-sequences-table-tokens {
    overflow-x: auto;
  }

  .sae-info {
    display: flex;
    gap: 0.25em;
    align-items: center;
    max-width: 16em;
  }

  .sae-info span {
    font-weight: var(--font-medium);
  }
</style>
