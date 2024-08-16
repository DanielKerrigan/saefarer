<script lang="ts">
  import { scaleLinear } from "d3-scale";
  import Axis from "./Axis.svelte";
  import { range } from "d3-array";
  import { line as d3line, area as d3area } from "d3-shape";
  import { zip } from "d3-array";

  let {
    xs,
    ys,
    width,
    height,
    bandY0,
    bandY1,
    marginLeft = 0,
    marginTop = 0,
    marginRight = 0,
    marginBottom = 0,
    xAxisLabel = "",
    yAxisLabel = "",
  }: {
    xs: number[];
    ys: number[];
    width: number;
    height: number;
    bandY0?: number[];
    bandY1?: number[];
    marginLeft?: number;
    marginTop?: number;
    marginRight?: number;
    marginBottom?: number;
    xAxisLabel?: string;
    yAxisLabel?: string;
  } = $props();

  let x = $derived(
    scaleLinear()
      .domain([xs[0], xs[xs.length - 1]])
      .range([marginLeft, width - marginRight])
  );

  let y = $derived(
    scaleLinear()
      .domain([0, Math.max(...ys)])
      .range([height - marginBottom, marginTop])
      .nice()
  );

  let I = $derived(range(xs.length));

  let line = $derived(
    d3line<number>()
      .x((i) => x(xs[i]))
      .y((i) => y(ys[i]))
  );

  let area = $derived(
    d3area<number[]>()
      .x((d) => x(d[0]))
      .y0((d) => y(d[1]))
      .y1((d) => y(d[2]))
  );
</script>

<svg {width} {height}>
  <path d={line(I)} stroke="black" fill="none" />

  {#if bandY0 && bandY1}
    <path
      d={area(zip(xs, bandY0, bandY1))}
      stroke="var(--gray-1)"
      fill="none"
    />
  {/if}

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
    {marginTop}
    {marginRight}
    {marginBottom}
    {marginLeft}
    numTicks={5}
  />
</svg>
