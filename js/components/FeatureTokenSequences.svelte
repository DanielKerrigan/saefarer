<script lang="ts">
  import { detail_feature } from "../synced-state.svelte";
  import type { ScaleSequential } from "d3-scale";
  import TokenSequences from "./TokenSequences.svelte";

  let { color }: { color: ScaleSequential<string> } = $props();

  let chosenIntervalKey = $state(
    Object.keys(detail_feature.value.sequence_intervals)[0],
  );
  let seqInterval = $derived(
    detail_feature.value.sequence_intervals[chosenIntervalKey],
  );

  let wrapSequences = $state(false);
</script>

<div class="sae-sequence-container">
  <div class="sae-sequences-controls">
    <select bind:value={chosenIntervalKey}>
      {#each Object.keys(detail_feature.value.sequence_intervals) as intervalName}
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

  <TokenSequences
    {color}
    sequences={seqInterval.sequences}
    wrap={wrapSequences}
  />
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
</style>
