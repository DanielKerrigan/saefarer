<script lang="ts">
  import type { ScaleOrdinal } from "d3-scale";

  let {
    color,
    labels,
    fontSize = 16,
    title = "",
    titleFontWeight = 500,
  }: {
    color: ScaleOrdinal<number, string>;
    labels: string[];
    fontSize?: number;
    title?: string;
    titleFontWeight?: number | string;
  } = $props();
</script>

<div class="sae-color-legend" style:font-size="{fontSize}px">
  {#if title}
    <div class="sae-color-legend-title" style:font-weight={titleFontWeight}>
      {title}
    </div>
  {/if}
  <div class="sae-color-legend-swatches">
    {#each color.domain() as d}
      <div class="sae-color-legend-swatch">
        <div class="sae-color-legend-square" style:background={color(d)}></div>
        <div class="sae-color-legend-label">{labels[d]}</div>
      </div>
    {/each}
  </div>
</div>

<style>
  .sae-color-legend {
    display: flex;
    gap: 0.5em;
    align-items: center;
  }

  .sae-color-legend-swatches {
    display: flex;
    gap: 0.5em;
  }

  .sae-color-legend-swatch {
    display: flex;
    gap: 0.25em;
    align-items: center;
  }

  .sae-color-legend-square {
    width: 1em;
    height: 1em;
  }
</style>
