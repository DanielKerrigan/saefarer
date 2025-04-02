<script lang="ts">
  import { scaleBand, scaleSequential } from "d3-scale";
  import { max } from "d3-array";
  import type { ConfusionMatrix, ConfusionMatrixCell } from "../../types";
  import Axis from "./axis/Axis.svelte";
  import { model_info } from "../../synced-state.svelte";
  import { interpolateYlGnBu } from "d3-scale-chromatic";
  import { rootDiv } from "../../state.svelte";
  import Tooltip from "../Tooltip.svelte";
  import ConfusionMatrixTooltip from "./ConfusionMatrixTooltip.svelte";
  import QuantitativeColorLegend from "./legends/QuantitativeColorLegend.svelte";

  let {
    cm,
    width,
    height,
    marginLeft = 72,
    marginTop = 72,
    marginRight = 72,
    marginBottom = 72,
    showLegend = true,
  }: {
    cm: ConfusionMatrix;
    width: number;
    height: number;
    marginLeft?: number;
    marginTop?: number;
    marginRight?: number;
    marginBottom?: number;
    showLegend?: boolean;
  } = $props();

  const legendGap = $derived(showLegend ? 4 : 0);
  const legendHeight = $derived(showLegend ? marginBottom - legendGap : 0);

  const x = $derived(
    scaleBand<number>()
      .domain(model_info.value.label_indices)
      .range([marginLeft, width - marginRight])
      .padding(0),
  );

  const y = $derived(
    scaleBand<number>()
      .domain(model_info.value.label_indices)
      .range([marginTop, height - marginBottom - legendGap])
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

  function onMouseEnterToken(event: MouseEvent, data: ConfusionMatrixCell) {
    if (!event.target || !rootDiv.value) {
      return;
    }

    const div = event.target as HTMLDivElement;
    const targetRect = div.getBoundingClientRect();
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

<div class="sae-cm-container">
  <svg {width} height={height - legendHeight}>
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
          onmouseenter={(event) => onMouseEnterToken(event, d)}
          onmouseleave={onMouseLeaveToken}
        />
      {/each}
    </g>

    <Axis
      orientation={"top"}
      scale={x}
      translateY={marginTop}
      title="Predicted label"
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
      title="True label"
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

  {#if showLegend}
    <QuantitativeColorLegend
      {width}
      height={legendHeight}
      {color}
      marginTop={16}
      {marginRight}
      marginBottom={32}
      {marginLeft}
      title={"Instance count"}
    />
  {/if}

  {#if tooltipInfo}
    <Tooltip {...tooltipInfo}>
      {#snippet content()}
        {#if tooltipInfo}
          <ConfusionMatrixTooltip data={tooltipInfo.data} />
        {/if}
      {/snippet}
    </Tooltip>
  {/if}
</div>

<style>
  .sae-cm-container {
    min-height: 0;
    display: flex;
    flex-direction: column;
  }

  .sae-cm-cell:hover {
    stroke: var(--color-red-600);
  }
</style>
