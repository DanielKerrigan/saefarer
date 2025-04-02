<script lang="ts">
  import type { ScaleOrdinal } from "d3-scale";

  let {
    color,
    labels,
  }: {
    color: ScaleOrdinal<number, string>;
    labels: string[];
  } = $props();
</script>

<div class="color-legend">
  {#each color.domain() as d}
    <div class="color-legend-swatch">
      <div class="color-legend-square" style:background={color(d)}></div>
      <div class="color-legend-label">{labels[d]}</div>
    </div>
  {/each}
</div>

<style>
  .color-legend {
    display: flex;
    gap: 1em;
  }

  .color-legend-swatch {
    display: flex;
    gap: 0.5em;
    align-items: center;
  }

  .color-legend-square {
    width: 1em;
    height: 1em;
  }
</style>
