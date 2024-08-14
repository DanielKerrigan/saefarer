<script lang="ts">
  import { scaleLinear } from "d3-scale";
  import type { FeatureProjection } from "../../types";
  import Axis from "./Axis.svelte";
  import { range } from "d3-array";

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
    data: FeatureProjection;
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
      .domain([Math.min(...data.x), Math.max(...data.x)])
      .range([marginLeft, width - marginRight])
  );

  let y = $derived(
    scaleLinear()
      .domain([Math.min(...data.y), Math.max(...data.y)])
      .range([height - marginBottom, marginTop])
      .nice()
  );

  let I = $derived(range(data.x.length));
</script>

<svg {width} {height}>
  <g>
    {#each I as i}
      <circle cx={x(data.x[i])} cy={y(data.y[i])} r={2} fill={"black"}>
        <title>Feature {data.feature_id[i]}</title>
      </circle>
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
  />

  <Axis
    orientation={"left"}
    scale={y}
    translateX={marginLeft}
    title={yAxisLabel}
    titleAnchor="top"
    {marginTop}
    {marginRight}
    {marginBottom}
    {marginLeft}
  />
</svg>
