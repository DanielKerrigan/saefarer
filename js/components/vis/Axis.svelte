<!-- Axis component based on d3-axis -->

<script lang="ts" generics="Domain">
  import type { AxisScale } from "./axis";
  import Label from "./Label.svelte";

  type Orientation = "top" | "right" | "bottom" | "left";
  type TitleAnchor = "top" | "right" | "bottom" | "left" | "center";
  type TitleArrow = "up" | "left" | "right" | "down" | "auto" | "none";

  let {
    orientation,
    scale,
    translateX = 0,
    translateY = 0,
    marginLeft = 0,
    marginTop = 0,
    marginRight = 0,
    marginBottom = 0,
    tickLineSize = 6,
    tickLabelFontSize = 10,
    tickPadding = 3,
    tickFormat,
    numTicks,
    tickValues,
    showTickMarks = true,
    showTickLabels = true,
    maxTickLabelSpace,
    tickLineColor = "black",
    tickLabelColor = "black",
    title = "",
    titleFontSize = 12,
    titleAnchor = "center",
    titleColor = "black",
    titleArrow = "auto",
  }: {
    orientation: Orientation;
    scale: AxisScale<Domain>;
    translateX?: number;
    translateY?: number;
    marginLeft?: number;
    marginTop?: number;
    marginRight?: number;
    marginBottom?: number;
    tickLineSize?: number;
    tickLabelFontSize?: number;
    tickPadding?: number;
    tickFormat?: (value: Domain) => string;
    numTicks?: number;
    tickValues?: Domain[];
    showTickMarks?: boolean;
    showTickLabels?: boolean;
    maxTickLabelSpace?: number;
    tickLineColor?: string;
    tickLabelColor?: string;
    title?: string;
    titleFontSize?: number;
    titleAnchor?: TitleAnchor;
    titleColor?: string;
    titleArrow?: TitleArrow;
  } = $props();

  let k = $derived(orientation === "top" || orientation === "left" ? -1 : 1);
  let tickSpacing = $derived(Math.max(tickLineSize, 0) + tickPadding);
  let offset = $derived(scale.bandwidth ? scale.bandwidth() / 2 : 0);

  let values = $derived(
    tickValues ?? (scale.ticks ? scale.ticks(numTicks) : scale.domain())
  );

  let format = $derived(
    tickFormat ??
      (scale.tickFormat
        ? scale.tickFormat(numTicks)
        : (d: Domain) => String(d).toString())
  );

  function getTitleWithArrow(
    title: string,
    titleArrow: TitleArrow,
    orientation: string,
    scale: AxisScale<Domain>
  ): string {
    if (titleArrow === "none" || !title || "bandwidth" in scale) {
      return title;
    }

    const arrows = {
      up: "↑",
      right: "→",
      down: "↓",
      left: "←",
    };

    if (titleArrow !== "auto") {
      return title + " " + arrows[titleArrow];
    }

    if (orientation === "left" || orientation === "right") {
      return title + " " + arrows.up;
    } else {
      return title + " " + arrows.right;
    }
  }

  let titleWithArrow = $derived(
    getTitleWithArrow(title, titleArrow, orientation, scale)
  );

  function getTitleLocation(
    orientation: Orientation,
    titleAnchor: TitleAnchor,
    scale: AxisScale<Domain>,
    marginLeft: number,
    marginTop: number,
    marginRight: number,
    marginBottom: number,
    fontSize: number
  ): { textAnchor: string; dy: string; x: number; y: number } {
    const minRange = Math.min(...scale.range());
    const maxRange = Math.max(...scale.range());
    const midRange = (minRange + maxRange) / 2;

    if (orientation === "left" || orientation == "right") {
      const textAnchor = orientation === "left" ? "start" : "end";
      const x = orientation === "left" ? -marginLeft : marginRight;
      if (titleAnchor === "top") {
        return { textAnchor, dy: "0.71em", x, y: minRange - marginTop };
      } else if (titleAnchor === "bottom") {
        return { textAnchor, dy: "0em", x, y: maxRange + marginBottom };
      } else {
        return { textAnchor, dy: "0.32em", x, y: midRange };
      }
    } else {
      const y =
        orientation === "top"
          ? -marginTop + fontSize / 2
          : marginBottom - fontSize / 2;
      const dy = orientation === "top" ? "0.71em" : "0em";
      if (titleAnchor === "left") {
        return {
          textAnchor: "start",
          dy,
          x: minRange - marginLeft,
          y,
        };
      } else if (titleAnchor === "right") {
        return { textAnchor: "end", dy, x: maxRange + marginRight, y };
      } else {
        return {
          textAnchor: "middle",
          dy,
          x: midRange,
          y,
        };
      }
    }
  }

  let titleLocation = $derived(
    getTitleLocation(
      orientation,
      titleAnchor,
      scale,
      marginLeft,
      marginTop,
      marginRight,
      marginBottom,
      titleFontSize
    )
  );
</script>

<g transform="translate({translateX},{translateY})">
  <g>
    {#each values as d}
      {#if orientation === "left" || orientation == "right"}
        {@const y = (scale(d) ?? 0) + offset}
        <g transform="translate(0,{y})">
          {#if showTickMarks}
            <line
              x1={tickLineSize * k}
              y1={0}
              x2={0}
              y2={0}
              stroke={tickLineColor}
            />
          {/if}
          {#if showTickLabels}
            {#if maxTickLabelSpace}
              <Label
                label={format(d)}
                width={maxTickLabelSpace}
                x={tickSpacing * k}
                y={0}
                dominantBaseline={"middle"}
                textAnchor={orientation === "left" ? "end" : "start"}
                fontSize={tickLabelFontSize}
                fontColor={tickLabelColor}
              />
            {:else}
              <text
                x={tickSpacing * k}
                y={0}
                dominant-baseline={"middle"}
                text-anchor={orientation === "left" ? "end" : "start"}
                font-size={tickLabelFontSize}
                fill={tickLabelColor}>{format(d)}</text
              >
            {/if}
          {/if}
        </g>
      {:else}
        {@const x = (scale(d) ?? 0) + offset}
        <g transform="translate({x},0)">
          {#if showTickMarks}
            <line
              x1={0}
              y1={tickLineSize * k}
              x2={0}
              y2={0}
              stroke={tickLineColor}
            />
          {/if}
          {#if showTickLabels}
            {#if maxTickLabelSpace}
              <Label
                label={format(d)}
                width={maxTickLabelSpace}
                x={0}
                y={tickSpacing * k}
                dominantBaseline={orientation === "top"
                  ? "text-top"
                  : "hanging"}
                textAnchor={"middle"}
                fontSize={tickLabelFontSize}
                fontColor={tickLabelColor}
              />
            {:else}
              <text
                x={0}
                y={tickSpacing * k}
                dominant-baseline={orientation === "top"
                  ? "text-top"
                  : "hanging"}
                text-anchor={"middle"}
                font-size={tickLabelFontSize}
                fill={tickLabelColor}>{format(d)}</text
              >
            {/if}
          {/if}
        </g>
      {/if}
    {/each}
  </g>

  {#if title}
    <text
      fill={titleColor}
      font-size={titleFontSize}
      text-anchor={titleLocation.textAnchor}
      dy={titleLocation.dy}
      y={titleLocation.y}
      x={titleLocation.x}>{titleWithArrow}</text
    >
  {/if}
</g>
