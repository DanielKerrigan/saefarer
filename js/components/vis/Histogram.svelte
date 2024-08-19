<script lang="ts">
  import { scaleLinear } from "d3-scale";
  import type { Histogram } from "../../types";
  import Axis from "./Axis.svelte";
  import { pairs, range } from "d3-array";
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
    data: Histogram;
    width: number;
    height: number;
    marginLeft?: number;
    marginTop?: number;
    marginRight?: number;
    marginBottom?: number;
    xAxisLabel?: string;
    yAxisLabel?: string;
  } = $props();

  let x = $derived(
    scaleLinear()
      .domain([data.thresholds[0], data.thresholds[data.thresholds.length - 1]])
      .range([marginLeft, width - marginRight])
  );

  let y = $derived(
    scaleLinear()
      .domain([0, Math.max(...data.counts)])
      .range([height - marginBottom, marginTop])
      .nice()
  );

  let I = $derived(range(data.counts.length));

  let edges = $derived(pairs(data.thresholds));
</script>

<svg {width} {height}>
  <g>
    {#each I as i}
      <rect
        x={x(edges[i][0]) + 0.5}
        width={x(edges[i][1]) - x(edges[i][0]) - 1}
        y={y(data.counts[i])}
        height={y(0) - y(data.counts[i])}
        fill={"black"}
      />
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
