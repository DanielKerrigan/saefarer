<script lang="ts">
  import { hcl } from "d3-color";
  import { scaleSequential } from "d3-scale";
  import type { FeatureData } from "../types";
  import { interpolateBlues } from "d3-scale-chromatic";

  let { feature }: { feature: FeatureData } = $props();

  const color = $derived(
    scaleSequential()
      .domain([0, feature.max_act])
      .interpolator(interpolateBlues),
  );

  const sequence = $derived(
    feature.sequence_intervals["Max Activations"].sequences[0],
  );

  const displayTokens = $derived(
    sequence.display_tokens.slice(
      Math.max(0, sequence.max_token_index - 5),
      Math.min(sequence.display_tokens.length, sequence.max_token_index + 5),
    ),
  );
</script>

<div class="sae-sequence">
  {#each displayTokens as dt, i}
    {@const col = color(dt.max_act)}
    <!-- TODO: do this properly -->
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div
      class="sae-token"
      style:background={col}
      style:color={hcl(col).l > 50
        ? "var(--color-black)"
        : "var(--color-white)"}
      style:font-weight={i === sequence.max_token_index ? "bold" : "normal"}
    >
      <span class="sae-token-name">{dt.display}</span>
    </div>
  {/each}
</div>

<style>
  .sae-sequence {
    display: flex;
    flex-wrap: wrap;
  }

  .sae-token-name {
    white-space: pre;
  }
</style>
