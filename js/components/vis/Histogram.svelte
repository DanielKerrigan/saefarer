<script lang="ts">
  import { scaleLinear } from "d3-scale";
  import type { HistogramData } from "../../types";
  import Axis from "./axis/Axis.svelte";
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
    showXAxis = true,
    showYAxis = true,
  }: {
    data: HistogramData;
    width: number;
    height: number;
    marginLeft?: number;
    marginTop?: number;
    marginRight?: number;
    marginBottom?: number;
    xAxisLabel?: string;
    yAxisLabel?: string;
    showXAxis?: boolean;
    showYAxis?: boolean;
  } = $props();

  let x = $derived(
    scaleLinear()
      .domain([data.thresholds[0], data.thresholds[data.thresholds.length - 1]])
      .range([marginLeft, width - marginRight]),
  );

  let y = $derived(
    scaleLinear()
      .domain([0, Math.max(...data.counts)])
      .range([height - marginBottom, marginTop])
      .nice(),
  );

  let I = $derived(range(data.counts.length));

  let edges = $derived(pairs(data.thresholds));
</script>

<svg {width} {height}>
  <g>
    {#each I as i}
      <rect
        x={x(edges[i][0]) + 0.5}
        width={Math.max(0, x(edges[i][1]) - x(edges[i][0]) - 1)}
        y={y(data.counts[i])}
        height={y(0) - y(data.counts[i])}
        fill={"var(--color-black)"}
      />
    {/each}
  </g>

  {#if showXAxis}
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
  {/if}

  {#if showYAxis}
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
  {/if}
</svg>
