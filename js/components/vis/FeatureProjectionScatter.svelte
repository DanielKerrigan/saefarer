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
      .domain([Math.min(...data.xs), Math.max(...data.xs)])
      .range([marginLeft, width - marginRight]),
  );

  let y = $derived(
    scaleLinear()
      .domain([Math.min(...data.ys), Math.max(...data.ys)])
      .range([height - marginBottom, marginTop])
      .nice(),
  );

  let I = $derived(range(data.xs.length));
</script>

<svg {width} {height}>
  <rect {width} {height} fill="var(--gray-0)" />
  <g>
    {#each I as i}
      <circle cx={x(data.xs[i])} cy={y(data.ys[i])} r={2} fill={"black"}>
        <title>Feature {data.feature_ids[i]}</title>
      </circle>
    {/each}
  </g>
</svg>
