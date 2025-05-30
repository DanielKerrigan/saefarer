<script lang="ts">
  import { scaleLinear } from "d3-scale";
  import type { HistogramData } from "../../types";
  import Axis from "./axis/Axis.svelte";
  import { pairs, range } from "d3-array";
  import { defaultFormat } from "./vis-utils";
  import { font_sizes } from "../../synced-state.svelte";
  import VisTooltip from "../VisTooltip.svelte";
  import TooltipTable from "../TooltipTable.svelte";

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
    xFormat = defaultFormat,
    yFormat = defaultFormat,
    tooltipEnabled = true,
    tooltipData = [
      {
        key: xAxisLabel,
        value: (x1, x2, _y) => `${xFormat(x1)} - ${xFormat(x2)}`,
      },
      {
        key: yAxisLabel,
        value: (_x1, _x2, y) => yFormat(y),
      },
    ],
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
    xFormat?: (x: number) => string;
    yFormat?: (x: number) => string;
    tooltipEnabled?: boolean;
    tooltipData?: {
      key: string;
      value: (x1: number, x2: number, y: number) => string;
    }[];
  } = $props();

  let x = $derived(
    scaleLinear()
      .domain([data.thresholds[0], data.thresholds[data.thresholds.length - 1]])
      .range([marginLeft, width - marginRight]),
  );

  let maxCount = $derived(Math.max(...data.counts));

  let y = $derived(
    scaleLinear()
      .domain([0, maxCount])
      .nice()
      .range([height - marginBottom, marginTop]),
  );

  let I = $derived(range(data.counts.length));

  let edges = $derived(pairs(data.thresholds));

  let tooltipInfo: {
    count: number;
    xMin: number;
    xMax: number;
    anchor: Element;
    index: number;
  } | null = $state(null);

  function onMouseEnter(
    event: MouseEvent & {
      currentTarget: EventTarget & SVGRectElement;
    },
    count: number,
    xMin: number,
    xMax: number,
    index: number,
  ) {
    tooltipInfo = {
      count,
      xMin,
      xMax,
      anchor: event.currentTarget,
      index,
    };
  }

  function onMouseLeave() {
    tooltipInfo = null;
  }
</script>

<div>
  <svg {width} {height}>
    <g>
      {#each I as i}
        <!-- background -->

        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <rect
          x={x(edges[i][0])}
          width={Math.max(0, x(edges[i][1]) - x(edges[i][0]))}
          y={y.range()[1]}
          height={y.range()[0] - y.range()[1]}
          fill={i === tooltipInfo?.index
            ? "var(--color-neutral-200)"
            : "var(--color-white)"}
          onmouseenter={tooltipEnabled
            ? (event) =>
                onMouseEnter(event, data.counts[i], edges[i][0], edges[i][1], i)
            : null}
          onmouseleave={tooltipEnabled ? onMouseLeave : null}
        />

        <!-- bar -->
        <rect
          style:pointer-events="none"
          x={x(edges[i][0]) + 0.5}
          width={Math.max(0, x(edges[i][1]) - x(edges[i][0]) - 1)}
          y={y(data.counts[i])}
          height={y(0) - y(data.counts[i])}
          fill={i === tooltipInfo?.index
            ? "var(--color-black)"
            : "var(--color-neutral-500)"}
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
        tickFormat={xFormat}
        {marginTop}
        {marginRight}
        {marginBottom}
        {marginLeft}
        numTicks={5}
        titleFontSize={font_sizes.sm}
        tickLabelFontSize={font_sizes.xs}
      />
    {/if}

    {#if showYAxis}
      <Axis
        orientation={"left"}
        scale={y}
        translateX={marginLeft}
        title={yAxisLabel}
        titleAnchor="top"
        tickFormat={yFormat}
        {marginTop}
        {marginRight}
        {marginBottom}
        {marginLeft}
        numTicks={5}
        titleFontSize={font_sizes.sm}
        tickLabelFontSize={font_sizes.xs}
      />
    {/if}
  </svg>

  {#if tooltipInfo}
    <VisTooltip {...tooltipInfo}>
      <TooltipTable
        data={tooltipData.map(({ key, value }) => ({
          key,
          value: value(
            tooltipInfo?.xMin ?? 0,
            tooltipInfo?.xMax ?? 0,
            tooltipInfo?.count ?? 0,
          ),
        }))}
      />
    </VisTooltip>
  {/if}
</div>

<style>
</style>
