<script lang="ts">
  import { scaleLinear, scaleOrdinal } from "d3-scale";
  import { schemeObservable10 } from "d3-scale-chromatic";
  import type { MarginalEffects } from "../../types";
  import Axis from "./axis/Axis.svelte";
  import { pairs, range } from "d3-array";
  import { line as d3line } from "d3-shape";
  import { defaultFormat } from "./vis-utils";
  import LabelColorLegend from "./legends/CategoricalColorLegend.svelte";
  import { model_info } from "../../synced-state.svelte";

  let {
    data,
    width,
    height,
    marginLeft = 0,
    marginTop = 0,
    marginRight = 0,
    marginBottom = 0,
    circleRadius = 2,
    xAxisLabel = "",
    yAxisLabel = "",
    showColorLegend = true,
    showXAxis = true,
    showYAxis = true,
  }: {
    data: MarginalEffects;
    width: number;
    height: number;
    marginLeft?: number;
    marginTop?: number;
    marginRight?: number;
    marginBottom?: number;
    circleRadius?: number;
    xAxisLabel?: string;
    yAxisLabel?: string;
    showColorLegend?: boolean;
    showXAxis?: boolean;
    showYAxis?: boolean;
  } = $props();

  const edges = $derived(pairs(data.thresholds));

  const x = $derived(
    scaleLinear()
      .domain([data.thresholds[0], data.thresholds[data.thresholds.length - 1]])
      .range([marginLeft, width - marginRight]),
  );

  const y = $derived(
    scaleLinear()
      .domain([0, Math.max(...data.probs.flat())])
      .range([height - marginBottom, marginTop])
      .nice(),
  );

  const line = $derived(
    d3line<number>()
      .x((d, i) => x((edges[i][0] + edges[i][1]) / 2))
      .y((d, i) => y(d))
      .defined((d) => d !== -1),
  );

  const color = $derived(
    scaleOrdinal<number, string>()
      .domain(range(data.probs.length))
      .range(schemeObservable10),
  );
</script>

<div>
  {#if showColorLegend}
    <LabelColorLegend {color} labels={model_info.value.labels} />
  {/if}

  <svg {width} {height}>
    <g>
      {#each data.probs as probs, labelIndex}
        <path
          d={line(probs)}
          stroke={color(labelIndex)}
          fill="none"
          stroke-linecap="round"
        />

        {#each probs as prob, binIndex}
          {#if prob !== -1}
            <circle
              cx={x((edges[binIndex][0] + edges[binIndex][1]) / 2)}
              cy={y(prob)}
              fill={color(labelIndex)}
              r={2}
            />
          {/if}
        {/each}
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
</div>
