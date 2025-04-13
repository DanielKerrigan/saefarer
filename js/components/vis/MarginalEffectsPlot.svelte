<script lang="ts">
  import { scaleLinear } from "d3-scale";
  import type { ScaleOrdinal } from "d3-scale";
  import type { HistogramData, MarginalEffectsData } from "../../types";
  import Axis from "./axis/Axis.svelte";
  import { pairs, zip } from "d3-array";
  import { line as d3line } from "d3-shape";
  import { defaultFormat } from "./vis-utils";
  import CategoricalColorLegend from "./legends/CategoricalColorLegend.svelte";
  import { model_info } from "../../synced-state.svelte";
  import Histogram from "./Histogram.svelte";

  let {
    marginalEffects,
    width,
    height,
    color,
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
    showBaseValues = true,
    baseValueWidth = 8,
    baseValuePadding = 4,
  }: {
    marginalEffects: MarginalEffectsData;
    width: number;
    height: number;
    color: ScaleOrdinal<number, string>;
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
    showBaseValues?: boolean;
    baseValueWidth?: number;
    baseValuePadding?: number;
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
      points: zip(binCenters, probsForLabel)
        .filter(([, prob]) => prob !== -1)
        .map(([act, prob]) => ({ act, prob })),
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

  const xAxisPadding = $derived(0);
  const yAxisPadding = $derived(
    showBaseValues ? baseValueWidth + baseValuePadding : 0,
  );
</script>

<div>
  {#if showColorLegend}
    <CategoricalColorLegend {color} labels={model_info.value.labels} />
  {/if}

  <svg {width} {height}>
    {#if distribution}
      <Histogram
        data={distribution}
        marginTop={0}
        {marginRight}
        {marginLeft}
        marginBottom={0}
        {width}
        height={marginTop}
        showXAxis={false}
        showYAxis={false}
      />
    {/if}

    {#if showXAxis}
      <Axis
        orientation={"bottom"}
        scale={x}
        translateY={height - marginBottom + xAxisPadding}
        title={xAxisLabel}
        titleAnchor="right"
        {marginTop}
        {marginRight}
        marginBottom={marginBottom - xAxisPadding}
        {marginLeft}
        numTicks={5}
        showDomain={true}
      />
    {/if}

    {#if showYAxis}
      <Axis
        orientation={"left"}
        scale={y}
        translateX={marginLeft - yAxisPadding}
        title={yAxisLabel}
        titleAnchor="center"
        tickFormat={defaultFormat}
        {marginTop}
        {marginRight}
        {marginBottom}
        marginLeft={marginLeft - yAxisPadding}
        numTicks={5}
      />
    {/if}

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

    {#if showBaseValues}
      <g
        transform="translate({marginLeft - baseValueWidth - baseValuePadding})"
      >
        <rect
          width={baseValueWidth}
          y={marginTop}
          height={height - marginTop - marginBottom}
          fill={"var(--color-neutral-100)"}
        />

        {#each model_info.value.label_indices as labelIndex}
          <circle
            cx={baseValueWidth / 2}
            cy={y(model_info.value.cm.pred_label_pcts[labelIndex])}
            fill={color(labelIndex)}
            fill-opacity={0.5}
            stroke={color(labelIndex)}
            r={baseValueWidth / 2 - 1}
          />
        {/each}

        <g
          transform="translate({baseValueWidth / 2},{height -
            marginBottom +
            xAxisPadding})"
        >
          <line y2="6" stroke="black" />
          <text
            dominant-baseline="hanging"
            text-anchor="end"
            font-size="10"
            font-family="ui-sans-serif, system-ui, sans-serif"
            y="9"
          >
            Baseline
          </text>
        </g>
      </g>
    {/if}
  </svg>
</div>
