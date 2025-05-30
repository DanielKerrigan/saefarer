<script lang="ts">
  import {
    scaleLinear,
    scaleSequential,
    scaleBand,
    scaleDiverging,
  } from "d3-scale";
  import type { HistogramData, MarginalEffectsData } from "../../types";
  import Axis from "./axis/Axis.svelte";
  import { pairs } from "d3-array";
  import { format } from "d3-format";
  import { interpolateReds, interpolatePRGn } from "d3-scale-chromatic";
  import {
    activationValueFormat,
    countFormat,
    defaultFormat,
    probabilityFormat,
  } from "./vis-utils";
  import QuantitativeColorLegend from "./legends/QuantitativeColorLegend.svelte";
  import { font_sizes, model_info } from "../../synced-state.svelte";
  import Histogram from "./Histogram.svelte";
  import DashedOutlineRect from "./DashedOutlineRect.svelte";
  import VisTooltip from "../VisTooltip.svelte";
  import TooltipTable from "../TooltipTable.svelte";

  let {
    marginalEffects,
    width,
    height,
    classes,
    distribution = null,
    compareToBaseProbs = false,
    maxColorDomain = null,
    marginTop = 0,
    marginRight = 0,
    marginBottom = 0,
    marginLeft = 0,
    xAxisLabel = "",
    yAxisLabel = "",
    showColorLegend = true,
    showXAxis = true,
    showYAxis = true,
    tooltipEnabled = true,
  }: {
    marginalEffects: MarginalEffectsData;
    width: number;
    height: number;
    classes: number[];
    distribution?: HistogramData | null;
    compareToBaseProbs?: boolean;
    maxColorDomain?: number | null;
    marginTop?: number;
    marginRight?: number;
    marginBottom?: number;
    marginLeft?: number;
    xAxisLabel?: string;
    yAxisLabel?: string;
    showColorLegend?: boolean;
    showXAxis?: boolean;
    showYAxis?: boolean;
    tooltipEnabled?: boolean;
  } = $props();

  const legendGap = 16;
  const legendWidth = $derived(showColorLegend ? marginRight - legendGap : 0);
  const legendHeight = $derived(showColorLegend ? height : 0);

  const histogramHeight = $derived(distribution ? marginTop - 2 : 0);

  const svgMarginTop = $derived(marginTop - histogramHeight);
  const svgMarginRight = $derived(marginRight - legendWidth);
  const svgWidth = $derived(width - legendWidth);
  const svgHeight = $derived(height - histogramHeight);

  const legendMarginTop = $derived(showColorLegend ? marginTop : 0);
  const legendMarginRight = $derived(showColorLegend ? 60 : 0);
  const legendMarginBottom = $derived(showColorLegend ? marginBottom : 0);
  const legendMarginLeft = $derived(0);

  type Point = {
    startAct: number;
    endAct: number;
    prob: number;
    delta: number;
  };

  type Series = {
    labelIndex: number;
    points: Point[];
  };

  const series: Series[] = $derived(
    marginalEffects.probs.map((probsForLabel, labelIndex) => ({
      labelIndex,
      points: pairs(marginalEffects.thresholds).map(([binStart, binEnd], i) => {
        const prob = probsForLabel[i] >= 0 ? probsForLabel[i] : NaN;
        const delta = Number.isNaN(prob)
          ? NaN
          : prob - model_info.value.cm.pred_label_pcts[labelIndex];
        return {
          startAct: binStart,
          endAct: binEnd,
          prob,
          delta,
        };
      }),
    })),
  );

  const x = $derived(
    scaleLinear()
      .domain([
        marginalEffects.thresholds[0],
        marginalEffects.thresholds[marginalEffects.thresholds.length - 1],
      ])
      .range([marginLeft, svgWidth - svgMarginRight]),
  );

  const y = $derived(
    scaleBand<number>()
      .domain(classes)
      .range([svgMarginTop, svgHeight - marginBottom]),
  );

  const maxProb = $derived(
    maxColorDomain ??
      Math.max(
        ...series.flatMap((s) =>
          s.points.map((p) => (Number.isNaN(p.prob) ? 0 : p.prob)),
        ),
      ),
  );

  const maxDelta = $derived(
    maxColorDomain ??
      Math.max(
        ...series.flatMap((s) =>
          s.points.map((p) => Math.abs(Number.isNaN(p.delta) ? 0 : p.delta)),
        ),
      ),
  );

  const sequentialColor = $derived(
    scaleSequential<string>()
      .domain([0, maxProb])
      .interpolator(interpolateReds)
      .unknown("var(--color-neutral-300)"),
  );

  const divergingColor = $derived(
    scaleDiverging<string>()
      .domain([-maxDelta, 0, maxDelta])
      .interpolator(interpolatePRGn)
      .unknown("var(--color-neutral-300)"),
  );

  let tooltipInfo: {
    point: Point;
    anchor: Element;
    labelIndex: number;
    pointIndex: number;
  } | null = $state(null);

  function onMouseEnter(
    event: MouseEvent & {
      currentTarget: EventTarget & SVGRectElement;
    },
    point: Point,
    labelIndex: number,
    pointIndex: number,
  ) {
    tooltipInfo = {
      point,
      anchor: event.currentTarget,
      labelIndex,
      pointIndex,
    };
  }

  function onMouseLeave() {
    tooltipInfo = null;
  }
</script>

<div class="sae-heatmap-container">
  <div>
    {#if distribution}
      <Histogram
        data={distribution}
        marginTop={0}
        marginRight={svgMarginRight}
        {marginLeft}
        marginBottom={0}
        width={svgWidth}
        height={histogramHeight}
        showXAxis={false}
        showYAxis={false}
        xFormat={activationValueFormat}
        {tooltipEnabled}
        tooltipData={[
          {
            key: "Instance count",
            value: (_x1, _x2, y) => countFormat(y),
          },
          {
            key: "Activation value",
            value: (x1, x2, _y) =>
              `${activationValueFormat(x1)} to ${activationValueFormat(x2)}`,
          },
        ]}
      />
    {/if}
    <svg width={svgWidth} height={svgHeight}>
      {#if showXAxis}
        <Axis
          orientation={"bottom"}
          scale={x}
          translateY={svgHeight - marginBottom}
          title={xAxisLabel}
          marginTop={svgMarginTop}
          marginRight={svgMarginRight}
          {marginBottom}
          {marginLeft}
          numTicks={5}
          tickLabelFontSize={font_sizes.xs}
          titleFontSize={font_sizes.sm}
        />
      {/if}
      {#if showYAxis}
        <Axis
          orientation={"left"}
          scale={y}
          translateX={marginLeft}
          tickFormat={(labelIndex) => model_info.value.labels[labelIndex]}
          title={yAxisLabel}
          marginTop={svgMarginTop}
          marginRight={svgMarginRight}
          {marginBottom}
          {marginLeft}
          tickLabelFontSize={font_sizes.xs}
          titleFontSize={font_sizes.sm}
        />
      {/if}
      <g>
        {#each series as { points, labelIndex }}
          {#each points as p, pointIndex}
            <!-- TODO: do this properly -->
            <!-- svelte-ignore a11y_no_static_element_interactions -->
            <rect
              x={x(p.startAct) + 0.5}
              width={x(p.endAct) - x(p.startAct) - 1}
              y={(y(labelIndex) ?? 0) + 0.5}
              height={y.bandwidth() - 1}
              fill={compareToBaseProbs
                ? divergingColor(p.delta)
                : sequentialColor(p.prob)}
              onmouseenter={tooltipEnabled
                ? (event) => onMouseEnter(event, p, labelIndex, pointIndex)
                : null}
              onmouseleave={tooltipEnabled ? onMouseLeave : null}
            />
            {#if labelIndex === tooltipInfo?.labelIndex && pointIndex === tooltipInfo.pointIndex}
              <DashedOutlineRect
                x={x(p.startAct) + 0.5}
                width={x(p.endAct) - x(p.startAct) - 1}
                y={(y(labelIndex) ?? 0) + 0.5}
                height={y.bandwidth() - 1}
              />
            {/if}
          {/each}
        {/each}
      </g>
    </svg>
  </div>

  {#if showColorLegend}
    <QuantitativeColorLegend
      width={legendWidth}
      height={legendHeight}
      color={compareToBaseProbs ? divergingColor : sequentialColor}
      orientation={"vertical"}
      marginTop={legendMarginTop}
      marginRight={legendMarginRight}
      marginBottom={legendMarginBottom}
      marginLeft={legendMarginLeft}
      title={compareToBaseProbs
        ? "Difference from base prob."
        : "Mean predicted probability"}
      tickLabelFontSize={font_sizes.xs}
      titleFontSize={font_sizes.sm}
      tickFormat={defaultFormat}
    />
  {/if}

  {#if tooltipInfo}
    <VisTooltip {...tooltipInfo}>
      {#if tooltipInfo}
        <TooltipTable
          data={[
            {
              key: "Activation value",
              value: `${activationValueFormat(tooltipInfo.point.startAct)} to ${activationValueFormat(
                tooltipInfo.point.endAct,
              )}`,
            },
            {
              key: "Predicted label",
              value: model_info.value.labels[tooltipInfo.labelIndex],
            },
            {
              key: "Mean probability",
              value: Number.isNaN(tooltipInfo.point.prob)
                ? "No data"
                : probabilityFormat(tooltipInfo.point.prob),
            },
            ...(compareToBaseProbs
              ? [
                  {
                    key: "Diff. from base prob.",
                    value: probabilityFormat(tooltipInfo.point.delta),
                  },
                ]
              : []),
          ]}
        />
      {/if}
    </VisTooltip>
  {/if}
</div>

<style>
  .sae-heatmap-container {
    min-height: 0;
    display: flex;
  }
</style>
