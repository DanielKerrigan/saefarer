<script lang="ts">
  import { scaleBand, scaleSequential } from "d3-scale";
  import { max } from "d3-array";
  import type { ConfusionMatrixData, ConfusionMatrixCell } from "../../types";
  import Axis from "./axis/Axis.svelte";
  import { model_info } from "../../synced-state.svelte";
  import { interpolateYlGnBu } from "d3-scale-chromatic";
  import { rootDiv } from "../../state.svelte";
  import Tooltip from "../Tooltip.svelte";
  import ConfusionMatrixTooltipContent from "./ConfusionMatrixTooltipContent.svelte";
  import QuantitativeColorLegend from "./legends/QuantitativeColorLegend.svelte";

  let {
    cm,
    width,
    height,
    marginTop = 72,
    marginRight = 72,
    marginBottom = 72,
    marginLeft = 72,
    legend = "horizontal",
  }: {
    cm: ConfusionMatrixData;
    width: number;
    height: number;
    marginTop?: number;
    marginRight?: number;
    marginBottom?: number;
    marginLeft?: number;
    legend?: "horizontal" | "vertical" | "none";
  } = $props();

  function getDimensions(
    width: number,
    height: number,
    marginTop: number,
    marginRight: number,
    marginBottom: number,
    marginLeft: number,
    legend: "horizontal" | "vertical" | "none",
  ): {
    svgWidth: number;
    svgHeight: number;
    legendWidth: number;
    legendHeight: number;
    legendMarginTop: number;
    legendMarginRight: number;
    legendMarginBottom: number;
    legendMarginLeft: number;
    xRange: [number, number];
    yRange: [number, number];
  } {
    const legendGap = 4;
    if (legend === "none") {
      return {
        svgWidth: width,
        svgHeight: height,
        legendWidth: 0,
        legendHeight: 0,
        legendMarginTop: 0,
        legendMarginRight: 0,
        legendMarginBottom: 0,
        legendMarginLeft: 0,
        xRange: [marginLeft, width - marginRight],
        yRange: [marginTop, height - marginBottom],
      };
    } else if (legend === "horizontal") {
      const legendHeight = marginBottom - legendGap;
      return {
        svgWidth: width,
        svgHeight: height - legendHeight,
        legendWidth: width,
        legendHeight: legendHeight,
        legendMarginTop: 16,
        legendMarginRight: marginRight,
        legendMarginBottom: 32,
        legendMarginLeft: marginLeft,
        xRange: [marginLeft, width - marginRight],
        yRange: [marginTop, height - marginBottom - legendGap],
      };
    } else {
      const legendWidth = marginRight - legendGap;
      return {
        svgWidth: width - legendWidth,
        svgHeight: height,
        legendWidth: legendWidth,
        legendHeight: height,
        legendMarginTop: marginTop,
        legendMarginRight: 48,
        legendMarginBottom: marginBottom,
        legendMarginLeft: 0,
        xRange: [marginLeft, width - marginRight - legendGap],
        yRange: [marginTop, height - marginBottom],
      };
    }
  }

  const dim = $derived(
    getDimensions(
      width,
      height,
      marginTop,
      marginRight,
      marginBottom,
      marginLeft,
      legend,
    ),
  );

  const x = $derived(
    scaleBand<number>()
      .domain(model_info.value.label_indices)
      .range(dim.xRange)
      .padding(0),
  );

  const y = $derived(
    scaleBand<number>()
      .domain(model_info.value.label_indices)
      .range(dim.yRange)
      .padding(0),
  );

  const color = $derived(
    scaleSequential<string>()
      .domain([0, max(cm.cells, (d) => d.count) ?? 0])
      .interpolator(interpolateYlGnBu),
  );

  function indexToLabel(i: number): string {
    return model_info.value.labels[i];
  }

  const tickLabelFontSize = 10;
  const tickPadding = 3;
  const tickLineSize = 6;

  const maxTickLabelSpaceTop = $derived(
    marginTop - tickLabelFontSize - tickPadding - tickLineSize,
  );
  const maxTickLabelSpaceLeft = $derived(
    marginLeft - tickLabelFontSize - tickPadding - tickLineSize,
  );

  let tooltipInfo: {
    data: ConfusionMatrixCell;
    rootRect: DOMRect;
    targetRect: DOMRect;
  } | null = $state(null);

  function onMouseEnterSquare(
    event: MouseEvent & {
      currentTarget: EventTarget & SVGRectElement;
    },
    data: ConfusionMatrixCell,
  ) {
    if (!rootDiv.value) {
      return;
    }

    const targetRect = event.currentTarget.getBoundingClientRect();
    const rootRect = rootDiv.value.getBoundingClientRect();

    tooltipInfo = {
      data,
      rootRect,
      targetRect,
    };
  }

  function onMouseLeaveToken() {
    tooltipInfo = null;
  }
</script>

<div
  class="sae-cm-container"
  style:flex-direction={legend === "vertical" ? "row" : "column"}
>
  <svg width={dim.svgWidth} height={dim.svgHeight}>
    <g>
      {#each cm.cells as d}
        <!-- TODO: do this properly -->
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <rect
          class="sae-cm-cell"
          x={x(d.pred_label)}
          width={x.bandwidth()}
          y={y(d.label)}
          height={y.bandwidth()}
          fill={color(d.count)}
          stroke={color(d.count)}
          stroke-width={2}
          clip-path="inset(1px)"
          onmouseenter={(event) => onMouseEnterSquare(event, d)}
          onmouseleave={onMouseLeaveToken}
        />
      {/each}
    </g>

    <Axis
      orientation={"top"}
      scale={x}
      translateY={marginTop}
      title="Predicted label (ŷ)"
      titleAnchor="center"
      tickFormat={indexToLabel}
      tickLabelAngle={maxTickLabelSpaceLeft <= x.bandwidth() ? 0 : -45}
      {marginTop}
      {marginRight}
      {marginBottom}
      {marginLeft}
      {tickLabelFontSize}
      {tickPadding}
      {tickLineSize}
      maxTickLabelSpace={maxTickLabelSpaceTop}
    />

    <Axis
      orientation={"left"}
      scale={y}
      translateX={marginLeft}
      title="True label (y)"
      titleAnchor="center"
      tickFormat={indexToLabel}
      {marginTop}
      {marginRight}
      {marginBottom}
      {marginLeft}
      {tickLabelFontSize}
      {tickPadding}
      {tickLineSize}
      maxTickLabelSpace={maxTickLabelSpaceTop}
    />
  </svg>

  {#if legend !== "none"}
    <QuantitativeColorLegend
      width={dim.legendWidth}
      height={dim.legendHeight}
      {color}
      orientation={legend}
      marginTop={dim.legendMarginTop}
      marginRight={dim.legendMarginRight}
      marginBottom={dim.legendMarginBottom}
      marginLeft={dim.legendMarginLeft}
      title={"Instance count"}
    />
  {/if}

  {#if tooltipInfo}
    <Tooltip {...tooltipInfo}>
      {#snippet content()}
        {#if tooltipInfo}
          <ConfusionMatrixTooltipContent data={tooltipInfo.data} />
        {/if}
      {/snippet}
    </Tooltip>
  {/if}
</div>

<style>
  .sae-cm-container {
    min-height: 0;
    display: flex;
  }

  .sae-cm-cell:hover {
    stroke: var(--color-red-600);
  }
</style>
