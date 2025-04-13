<script lang="ts">
  import { base_font_size, detail_feature } from "../synced-state.svelte";
  import type { ScaleSequential, ScaleOrdinal } from "d3-scale";
  import TokenSequence from "./TokenSequence.svelte";

  let {
    labelColor,
    tokenColor,
  }: {
    labelColor: ScaleOrdinal<number, string>;
    tokenColor: ScaleSequential<string>;
  } = $props();

  let chosenIntervalKey = $state(
    Object.keys(detail_feature.value.sequence_intervals)[0],
  );
  let seqInterval = $derived(
    detail_feature.value.sequence_intervals[chosenIntervalKey],
  );

  const cellPaddingX = $derived(base_font_size.value * 0.5);
  const cellPaddingY = $derived(base_font_size.value * 0.25);

  let wrapSequences = $state(false);
</script>

<div class="sae-sequence-container">
  <div class="sae-sequences-controls">
    <label>
      <span>Example Activations:</span>
      <select bind:value={chosenIntervalKey}>
        {#each Object.keys(detail_feature.value.sequence_intervals) as intervalName}
          <option value={intervalName}>
            {intervalName}
          </option>
        {/each}
      </select>
    </label>

    <label>
      <input type="checkbox" bind:checked={wrapSequences} />
      <span>Wrap</span>
    </label>
  </div>

  <div
    class="sae-sequences-table"
    style:--cell-padding-x="{cellPaddingX}px"
    style:--cell-padding-y="{cellPaddingY}px"
  >
    <div
      class="sae-sequences-table-cell sae-sequences-table-header sae-sequences-table-number-header"
    >
      #
    </div>
    <div
      class="sae-sequences-table-cell sae-sequences-table-header sae-sequences-table-square-header"
    >
      ŷ
    </div>
    <div
      class="sae-sequences-table-cell sae-sequences-table-header sae-sequences-table-square-header"
    >
      y
    </div>
    <div class="sae-sequences-table-cell sae-sequences-table-header">
      TOKENS
    </div>

    {#each seqInterval.sequences as seq, i}
      {@const showBorder = i !== seqInterval.sequences.length - 1}
      <div
        class="sae-sequences-table-cell sae-sequences-table-number-value"
        class:sae-sequences-table-border={showBorder}
      >
        {seq.sequence_index}
      </div>
      <div
        class="sae-sequences-table-cell sae-sequences-table-y"
        class:sae-sequences-table-border={showBorder}
      >
        <div
          class="sae-sequences-table-square"
          style:background-color={labelColor(seq.pred_label)}
        ></div>
      </div>
      <div
        class="sae-sequences-table-cell sae-sequences-table-y"
        class:sae-sequences-table-border={showBorder}
      >
        <div
          class="sae-sequences-table-square"
          style:background-color={labelColor(seq.label)}
        ></div>
      </div>
      <div
        class="sae-sequences-table-cell sae-sequences-table-tokens"
        class:sae-sequences-table-border={showBorder}
      >
        <TokenSequence color={tokenColor} sequence={seq} wrap={wrapSequences} />
      </div>
    {/each}
  </div>
</div>

<style>
  select {
    align-self: flex-start;
    border: 1px solid var(--color-black);
    border-radius: 0.25em;
  }

  label {
    display: flex;
    align-items: center;
    gap: 0.5em;
  }

  label span:first-child {
    font-weight: 500;
  }

  .sae-sequence-container {
    min-height: 0;
    max-height: 100%;
    display: flex;
    flex-direction: column;
    gap: 0.5em;
  }

  .sae-sequences-controls {
    display: flex;
    justify-content: space-between;
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
    padding: var(--cell-padding-y) var(--cell-padding-x);
  }

  .sae-sequences-table-header {
    font-weight: 500;
    position: sticky;
    top: 0;
    z-index: 10;
    background-color: var(--color-white);
  }

  .sae-sequences-table-number-header {
    text-align: end;
  }

  .sae-sequences-table-number-value {
    font-family: var(--font-mono);
    display: flex;
    justify-content: end;
    align-items: center;
  }

  .sae-sequences-table-square-header {
    text-align: center;
  }

  .sae-sequences-table-y {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .sae-sequences-table-square {
    justify-self: center;
    align-self: center;
    width: 1em;
    height: 1em;
  }

  .sae-sequences-table-border {
    border-bottom: 1px solid var(--color-neutral-300);
  }

  .sae-sequences-table-tokens {
    overflow-x: auto;
  }
</style>
