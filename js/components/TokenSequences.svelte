<script lang="ts">
  import type { ScaleSequential } from "d3-scale";
  import type { FeatureTokenSequence } from "../types";
  import TokenSequence from "./TokenSequence.svelte";

  let {
    color,
    sequences,
    wrap,
  }: {
    color: ScaleSequential<string>;
    sequences: FeatureTokenSequence[];
    wrap: boolean;
  } = $props();
</script>

<div class="sae-sequences">
  {#each sequences as sequence}
    <div class="sae-sequence">
      <TokenSequence {color} {sequence} {wrap} />
    </div>
  {/each}
</div>

<style>
  .sae-sequences {
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    gap: 0.25em;
  }

  .sae-sequence + .sae-sequence {
    padding-top: 0.25em;
    border-top: 2px solid var(--color-neutral-100);
  }
</style>
