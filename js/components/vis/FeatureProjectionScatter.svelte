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
  }: {
    data: FeatureProjection;
    width: number;
    height: number;
    marginLeft?: number;
    marginTop?: number;
    marginRight?: number;
    marginBottom?: number;
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
  <rect {width} {height} fill="var(--gray-0)" />
  <g>
    {#each I as i}
      <circle cx={x(data.x[i])} cy={y(data.y[i])} r={2} fill={"black"}>
        <title>Feature {data.feature_id[i]}</title>
      </circle>
    {/each}
  </g>
</svg>
