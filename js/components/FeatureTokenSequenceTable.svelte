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
  import Info from "./icons/InfoIcon.svelte";
  import InfoIcon from "./icons/InfoIcon.svelte";

  let {
    tokenColor,
  }: {
    tokenColor: ScaleSequential<string>;
  } = $props();

  let chosenIntervalKey = $state(0);
  let seqInterval = $derived(
    detail_feature.value.sequence_intervals[chosenIntervalKey],
  );

  let wrapSequences = $state(false);
</script>

<div class="sae-sequence-container">
  <div class="sae-sequences-header">
    <div class="sae-sequences-controls">
      <span style:font-weight="var(--font-medium)">Example Activations</span>
      <label>
        <span>Range:</span>
        <select bind:value={chosenIntervalKey}>
          <option value={0}> Max activations </option>
          {#each range(detail_feature.value.sequence_intervals.length - 1, 0, -1) as i}
            <option value={i}>
              Interval {i}
            </option>
          {/each}
        </select>
      </label>
      <label>
        <input type="checkbox" bind:checked={wrapSequences} />
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
            <TooltipTable
              data={[{ key: "Instance index", value: `${seq.sequence_index}` }]}
            />
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
          wrap={wrapSequences}
          hidePadding={false}
        />
      </div>
    {/each}
  </div>
</div>

<style>
  select {
    border: 1px solid var(--color-black);
    border-radius: 0.25em;
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
</style>
