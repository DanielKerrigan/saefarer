<script lang="ts">
  import { hcl } from "d3-color";
  import { feature_data } from "../synced-state.svelte";
  import Tooltip from "./Tooltip.svelte";
  import type { ScaleSequential } from "d3-scale";
  import { rootDiv } from "../state.svelte";
  import type { FeatureToken, FeatureTokenSequence } from "../types";
  import FeatureTokenSequencesTooltip from "./FeatureTokenSequencesTooltip.svelte";

  let { color }: { color: ScaleSequential<string> } = $props();

  let chosenIntervalKey = $state(Object.keys(feature_data.value.sequence_intervals)[0]);
  let seqInterval = $derived(feature_data.value.sequence_intervals[chosenIntervalKey]);

  let wrapSequences = $state(false);


  let tooltipInfo: {
    data: FeatureToken;
    rootRect: DOMRect;
    targetRect: DOMRect;
  } | null = $state(null);

  function onMouseEnterToken(
    event: MouseEvent,
    sequence: FeatureTokenSequence,
    tokIndex: number
  ) {
    if (!event.target || !rootDiv.value) {
      return;
    }

    const div = event.target as HTMLDivElement;
    const targetRect = div.getBoundingClientRect();
    const rootRect = rootDiv.value.getBoundingClientRect();

    const data = {
      token: sequence.token[tokIndex],
      activation: sequence.activation[tokIndex],
      extras: Object.entries(sequence.extras).map(([key, values]) => ({ key, value: values[tokIndex]}))
    };

    tooltipInfo = {
      data,
      rootRect,
      targetRect
    }
  }

  function onMouseLeaveToken() {
    tooltipInfo = null;
  }
</script>

<div class="sae-sequence-container">
  <div class="sae-sequences-controls">
    <select bind:value={chosenIntervalKey}>
      {#each Object.keys(feature_data.value.sequence_intervals) as intervalName}
        <option value={intervalName}>
          {intervalName}
        </option>
      {/each}
    </select>

    <label>
      <input type="checkbox" bind:checked={wrapSequences} />
      <span>Wrap</span>
    </label>
  </div>

  <div class="sae-sequences">
    {#each seqInterval.sequences as seq}
      <div
        class="sae-sequence"
        style:flex-wrap={wrapSequences ? "wrap" : "nowrap"}
      >
        {#each seq.token as token, tokIndex}
          {@const col = color(seq.activation[tokIndex])}
          <!-- TODO: do this properly -->
          <!-- svelte-ignore a11y_no_static_element_interactions -->
          <div
            class="sae-token"
            style:background={col}
            style:color={hcl(col).l > 50 ? "black" : "white"}
            style:font-weight={tokIndex === seq.max_index ? "bold" : "normal"}
            style:--border-color={col}
            onmouseenter={(event) =>
              onMouseEnterToken(event, seq, tokIndex)}
            onmouseleave={onMouseLeaveToken}
          >
            <span class="sae-token-name">{token}</span>
          </div>
        {/each}
      </div>
    {/each}
  </div>

  {#if tooltipInfo}
    <Tooltip {...tooltipInfo}>
      {#snippet content()}
        {#if tooltipInfo}
          <FeatureTokenSequencesTooltip data={tooltipInfo.data} />
        {/if}
      {/snippet}
    </Tooltip>
  {/if}
</div>

<style>
  select {
    align-self: flex-start;
  }

  label {
    display: flex;
    align-items: center;
    gap: 0.25em;
  }

  .sae-sequence-container {
    min-height: 0;
    display: flex;
    flex-direction: column;
  }

  .sae-sequences-controls {
    display: flex;
    justify-content: space-between;
  }

  .sae-sequences {
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    gap: 0.25em;
  }

  .sae-sequence + .sae-sequence {
    padding-top: 0.25em;
    border-top: 2px solid var(--gray-1);
  }

  .sae-sequence {
    display: flex;
  }

  .sae-token-name {
    white-space: pre;
  }

  .sae-token {
    border: 1px solid var(--border-color);
  }

  .sae-token:hover {
    border-color: red;
  }
</style>
