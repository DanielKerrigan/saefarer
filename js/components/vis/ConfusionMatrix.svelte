<script lang="ts">
  import {
    scaleBand,
    scaleDiverging,
    scaleSequential,
    type ScaleDiverging,
    type ScaleSequential,
  } from "d3-scale";
  import { extent, max, zip } from "d3-array";
  import type { ConfusionMatrixData, ConfusionMatrixCell } from "../../types";
  import Axis from "./axis/Axis.svelte";
  import { font_sizes, model_info } from "../../synced-state.svelte";
  import { interpolateOranges, interpolatePiYG } from "d3-scale-chromatic";
  import QuantitativeColorLegend from "./legends/QuantitativeColorLegend.svelte";
  import VisTooltip from "../VisTooltip.svelte";
  import DashedOutlineRect from "./DashedOutlineRect.svelte";
  import {
    countFormat,
    percentagePointFormat,
    percentFormat,
  } from "./vis-utils";
  import TooltipTable from "../TooltipTable.svelte";

  let {
    cm,
    width,
    height,
    other,
    showDifference = false,
    marginTop = 72,
    marginRight = 72,
    marginBottom = 72,
    marginLeft = 72,
    legend = "horizontal",
  }: {
    cm: ConfusionMatrixData;
    width: number;
    height: number;
    other?: ConfusionMatrixData;
    showDifference?: boolean;
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
  } {
    const legendGap = 16;
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
      };
    } else {
      const legendWidth = marginRight - legendGap;
      return {
        svgWidth: width - legendWidth,
        svgHeight: height,
        legendWidth: legendWidth,
        legendHeight: height,
        legendMarginTop: marginTop,
        legendMarginRight: 60,
        legendMarginBottom: marginBottom,
        legendMarginLeft: 0,
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

  type CMCellDelta = ConfusionMatrixCell & { pp_delta: number };

  function getData(
    cm: ConfusionMatrixData,
    other: ConfusionMatrixData | undefined,
  ): CMCellDelta[] {
    if (other !== undefined) {
      return zip(cm.cells, other.cells).map(([subsetCell, wholeCell]) => ({
        ...subsetCell,
        pp_delta: subsetCell.pct - wholeCell.pct,
      }));
    }

    return cm.cells.map((d) => ({ ...d, pp_delta: 0 }));
  }

  const cells = $derived(getData(cm, other));

  const x = $derived(
    scaleBand<number>()
      .domain(model_info.value.label_indices)
      .range([marginLeft, width - marginRight])
      .padding(0),
  );

  const y = $derived(
    scaleBand<number>()
      .domain(model_info.value.label_indices)
      .range([marginTop, height - marginBottom])
      .padding(0),
  );

  function getColor(
    cells: CMCellDelta[],
    showDifference: boolean,
  ): ScaleSequential<string> | ScaleDiverging<string> {
    if (showDifference) {
      const [minDelta, maxDelta] = extent(cells, (d) => d.pp_delta);
      const absMax = Math.max(Math.abs(minDelta ?? 0), Math.abs(maxDelta ?? 0));

      return scaleDiverging<string>()
        .domain([-absMax, 0, absMax])
        .interpolator(interpolatePiYG);
    }

    return scaleSequential<string>()
      .domain([0, max(cells, (d) => d.pct) ?? 0])
      .interpolator(interpolateOranges);
  }

  const color = $derived(getColor(cells, showDifference));

  function indexToLabel(i: number): string {
    return model_info.value.labels[i];
  }

  const tickLabelFontSize = font_sizes.xs;
  const tickPadding = 3;
  const tickLineSize = 6;

  const maxTickLabelSpaceBottom = $derived(
    marginBottom - tickLabelFontSize - tickPadding - tickLineSize,
  );
  const maxTickLabelSpaceLeft = $derived(
    marginLeft - tickLabelFontSize - tickPadding - tickLineSize,
  );

  let tooltipInfo: {
    data: CMCellDelta;
    anchor: Element;
    index: number;
  } | null = $state(null);

  function onMouseEnterSquare(
    event: MouseEvent & {
      currentTarget: EventTarget & SVGRectElement;
    },
    data: CMCellDelta,
    index: number,
  ) {
    tooltipInfo = {
      data,
      anchor: event.currentTarget,
      index,
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
      {#each cells as d, i}
        {@const col = showDifference ? color(d.pp_delta) : color(d.pct)}
        <!-- TODO: do this properly -->
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <rect
          class="sae-cm-cell"
          x={(x(d.label) ?? 0) + 0.5}
          width={x.bandwidth() - 1}
          y={(y(d.pred_label) ?? 0) + 1}
          height={y.bandwidth() - 1}
          fill={col}
          onmouseenter={(event) => onMouseEnterSquare(event, d, i)}
          onmouseleave={onMouseLeaveToken}
        />

        {#if i === tooltipInfo?.index}
          <DashedOutlineRect
            x={(x(d.label) ?? 0) + 0.5}
            width={x.bandwidth() - 1}
            y={(y(d.pred_label) ?? 0) + 1}
            height={y.bandwidth() - 1}
          />
        {/if}
      {/each}
    </g>

    <Axis
      orientation={"bottom"}
      scale={x}
      translateY={dim.svgHeight - marginBottom}
      title="True label"
      titleAnchor="center"
      tickFormat={indexToLabel}
      tickLabelAngle={maxTickLabelSpaceBottom <= x.bandwidth() ? 0 : -45}
      {marginTop}
      {marginRight}
      {marginBottom}
      {marginLeft}
      {tickLabelFontSize}
      {tickPadding}
      {tickLineSize}
      maxTickLabelSpace={maxTickLabelSpaceBottom}
      titleFontSize={font_sizes.sm}
    />

    <Axis
      orientation={"left"}
      scale={y}
      translateX={marginLeft}
      title="Predicted label"
      titleAnchor="center"
      tickFormat={indexToLabel}
      {marginTop}
      {marginRight}
      {marginBottom}
      {marginLeft}
      {tickLabelFontSize}
      {tickPadding}
      {tickLineSize}
      maxTickLabelSpace={maxTickLabelSpaceLeft}
      titleFontSize={font_sizes.sm}
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
      title={showDifference ? "Percentage point difference" : "Percent of data"}
      {tickLabelFontSize}
      titleFontSize={font_sizes.sm}
      tickFormat={showDifference ? percentagePointFormat : percentFormat}
    />
  {/if}

  {#if tooltipInfo}
    <VisTooltip {...tooltipInfo}>
      {#if tooltipInfo}
        <TooltipTable
          data={[
            {
              key: "True label",
              value: model_info.value.labels[tooltipInfo.data.label],
            },
            {
              key: "Predicted label",
              value: model_info.value.labels[tooltipInfo.data.pred_label],
            },
            {
              key: "Percent of data",
              value: percentFormat(tooltipInfo.data.pct),
            },
            {
              key: "Instance count",
              value: countFormat(tooltipInfo.data.count),
            },
            ...(showDifference
              ? [
                  {
                    key: "Difference",
                    value: `${percentagePointFormat(tooltipInfo.data.pp_delta)} pp`,
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
  .sae-cm-container {
    min-height: 0;
    display: flex;
  }
</style>
