<script lang="ts">
  import { scaleLinear, scaleOrdinal } from "d3-scale";
  import { schemeObservable10 } from "d3-scale-chromatic";
  import type { MarginalEffects } from "../../types";
  import Axis from "./Axis.svelte";
  import { pairs, range } from "d3-array";
  import { line as d3line } from "d3-shape";
  import { defaultFormat } from "./vis-utils";

  let {
    data,
    width,
    height,
    marginLeft = 0,
    marginTop = 0,
    marginRight = 0,
    marginBottom = 0,
    xAxisLabel = "",
    yAxisLabel = "",
  }: {
    data: MarginalEffects;
    width: number;
    height: number;
    marginLeft?: number;
    marginTop?: number;
    marginRight?: number;
    marginBottom?: number;
    xAxisLabel?: string;
    yAxisLabel?: string;
  } = $props();

  const probabilities = $derived(
    data.probs.map((probs) =>
      probs.map((p) => (p.toString() === "nan" ? NaN : p)),
    ),
  );

  $effect(() => console.log("data", $state.snapshot(data)));
  $effect(() => console.log("probs", $state.snapshot(probabilities)));

  const edges = $derived(pairs(data.thresholds));

  const x = $derived(
    scaleLinear()
      .domain([data.thresholds[0], data.thresholds[data.thresholds.length - 1]])
      .range([marginLeft, width - marginRight]),
  );

  const y = $derived(
    scaleLinear()
      .domain([
        0,
        Math.max(...probabilities.flat().filter((d) => !Number.isNaN(d))),
      ])
      .range([height - marginBottom, marginTop])
      .nice(),
  );

  $effect(() => console.log("domain", y.domain()));

  const line = $derived(
    d3line<number>()
      .x((d, i) => x((edges[i][0] + edges[i][1]) / 2))
      .y((d, i) => y(d))
      .defined((d) => !Number.isNaN(d)),
  );

  const color = $derived(
    scaleOrdinal<number, string>()
      .domain(range(probabilities.length))
      .range(schemeObservable10),
  );
</script>

<div class="color-legend">
  {#each color.domain() as d}
    <div class="color-legend-swatch">
      <div class="color-legend-square" style:background={color(d)}></div>
      <div class="color-legend-label">{d}</div>
    </div>
  {/each}
</div>

<svg {width} {height}>
  <g>
    {#each probabilities as probs, i}
      <path d={line(probs)} stroke={color(i)} fill="none" />
    {/each}
  </g>

  <Axis
    orientation={"bottom"}
    scale={x}
    translateY={height - marginBottom}
    title={xAxisLabel}
    titleAnchor="right"
    {marginTop}
    {marginRight}
    {marginBottom}
    {marginLeft}
    numTicks={5}
  />
  <Axis
    orientation={"left"}
    scale={y}
    translateX={marginLeft}
    title={yAxisLabel}
    titleAnchor="top"
    tickFormat={defaultFormat}
    {marginTop}
    {marginRight}
    {marginBottom}
    {marginLeft}
    numTicks={5}
  />
</svg>

<style>
  .color-legend {
    display: flex;
    gap: 2em;
  }

  .color-legend-swatch {
    display: flex;
    gap: 1em;
    align-items: center;
  }

  .color-legend-square {
    width: 1em;
    height: 1em;
  }
</style>
