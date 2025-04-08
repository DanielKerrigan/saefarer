<script lang="ts">
  import { scaleLinear, scaleOrdinal } from "d3-scale";
  import { schemeObservable10 } from "d3-scale-chromatic";
  import type { HistogramData, MarginalEffectsData } from "../../types";
  import Axis from "./axis/Axis.svelte";
  import { pairs, zip } from "d3-array";
  import { line as d3line } from "d3-shape";
  import { defaultFormat } from "./vis-utils";
  import LabelColorLegend from "./legends/CategoricalColorLegend.svelte";
  import { model_info } from "../../synced-state.svelte";
  import Histogram from "./Histogram.svelte";

  let {
    marginalEffects,
    width,
    height,
    distribution = null,
    marginTop = 0,
    marginRight = 0,
    marginBottom = 0,
    marginLeft = 0,
    circleRadius = 2,
    xAxisLabel = "",
    yAxisLabel = "",
    showColorLegend = true,
    showXAxis = true,
    showYAxis = true,
  }: {
    marginalEffects: MarginalEffectsData;
    width: number;
    height: number;
    distribution?: HistogramData | null;
    marginTop?: number;
    marginRight?: number;
    marginBottom?: number;
    marginLeft?: number;
    circleRadius?: number;
    xAxisLabel?: string;
    yAxisLabel?: string;
    showColorLegend?: boolean;
    showXAxis?: boolean;
    showYAxis?: boolean;
  } = $props();

  type Point = { act: number; prob: number };
  type Series = {
    labelIndex: number;
    points: Point[];
  };

  const binCenters = $derived(
    pairs(marginalEffects.thresholds).map(
      ([binStart, binEnd]) => (binEnd + binStart) / 2,
    ),
  );

  const series: Series[] = $derived(
    marginalEffects.probs.map((probsForLabel, labelIndex) => ({
      labelIndex,
      points: [
        { act: 0, prob: marginalEffects.non_act_probs[labelIndex] },
        ...zip(binCenters, probsForLabel)
          .filter(([, prob]) => prob !== -1)
          .map(([act, prob]) => ({ act, prob })),
      ],
    })),
  );

  const x = $derived(
    scaleLinear()
      .domain([
        marginalEffects.thresholds[0],
        marginalEffects.thresholds[marginalEffects.thresholds.length - 1],
      ])
      .range([marginLeft, width - marginRight]),
  );

  const y = $derived(
    scaleLinear()
      .domain([0, Math.max(...marginalEffects.probs.flat())])
      .range([height - marginBottom, marginTop])
      .nice(),
  );

  const line = $derived(
    d3line<Point>()
      .x((d) => x(d.act))
      .y((d) => y(d.prob)),
  );

  const color = $derived(
    scaleOrdinal<number, string>()
      .domain(model_info.value.label_indices)
      .range(schemeObservable10),
  );
</script>

<div>
  {#if showColorLegend}
    <LabelColorLegend {color} labels={model_info.value.labels} />
  {/if}

  {#if distribution}
    <Histogram
      data={distribution}
      marginTop={0}
      {marginRight}
      {marginLeft}
      marginBottom={0}
      {width}
      height={64}
      showXAxis={false}
      showYAxis={false}
    />
  {/if}

  <svg {width} {height}>
    <g>
      {#each series as { points, labelIndex }}
        <path
          d={line(points)}
          stroke={color(labelIndex)}
          fill="none"
          stroke-linecap="round"
        />

        {#if circleRadius > 0}
          {#each points as p}
            {#if p.prob !== -1}
              <circle
                cx={x(p.act)}
                cy={y(p.prob)}
                fill={color(labelIndex)}
                r={circleRadius}
              />
            {/if}
          {/each}
        {/if}
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
        titleAnchor="center"
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
