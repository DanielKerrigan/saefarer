<!-- Axis component inspired by d3-axis and Observable Plot -->

<script lang="ts" generics="D extends Domain">
  import type { Scale, Orientation, TitleAnchor, Domain } from "./axis";
  import {
    getTitleLocation,
    textAlignToAnchor,
    getTickLabelTextAlign,
  } from "./axis";
  import Label from "./Label.svelte";

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
    tickLabelFontFamily = "ui-sans-serif, system-ui, sans-serif",
    tickLabelAngle = 0,
    tickPadding = 3,
    tickFormat,
    numTicks,
    tickValues,
    showTickMarks = true,
    showTickLabels = true,
    maxTickLabelSpace,
    tickLineColor = "black",
    tickLabelColor = "black",
    showDomain = false,
    domainColor = "black",
    title = "",
    titleFontSize = 12,
    titleFontFamily = "ui-sans-serif, system-ui, sans-serif",
    titleFontWeight = 400,
    titleAnchor = "center",
    titleOffsetX = 0,
    titleOffsetY = 0,
    titleColor = "black",
  }: {
    orientation: Orientation;
    scale: Scale<D>;
    translateX?: number;
    translateY?: number;
    marginLeft?: number;
    marginTop?: number;
    marginRight?: number;
    marginBottom?: number;
    tickLineSize?: number;
    tickLabelFontSize?: number;
    tickLabelFontFamily?: string;
    tickLabelAngle?: number;
    tickPadding?: number;
    tickFormat?: (value: D) => string;
    numTicks?: number;
    tickValues?: D[];
    showTickMarks?: boolean;
    showTickLabels?: boolean;
    maxTickLabelSpace?: number;
    tickLineColor?: string;
    tickLabelColor?: string;
    showDomain?: boolean;
    domainColor?: string;
    title?: string;
    titleFontSize?: number;
    titleFontFamily?: string;
    titleFontWeight?: number;
    titleOffsetX?: number;
    titleOffsetY?: number;
    titleAnchor?: TitleAnchor;
    titleColor?: string;
  } = $props();

  const k = $derived(orientation === "top" || orientation === "left" ? -1 : 1);
  const tickSpacing = $derived(Math.max(tickLineSize, 0) + tickPadding);
  const offset = $derived(scale.bandwidth ? scale.bandwidth() / 2 : 0);

  const titleLocation = $derived(
    getTitleLocation(
      orientation,
      titleAnchor,
      scale,
      marginLeft,
      marginTop,
      marginRight,
      marginBottom,
      titleFontSize,
    ),
  );

  let values = $derived(
    tickValues ?? (scale.ticks ? scale.ticks(numTicks) : scale.domain()),
  );

  let format = $derived(
    tickFormat ??
      (scale.tickFormat
        ? scale.tickFormat(numTicks)
        : (d: D) => String(d).toString()),
  );
</script>

<g transform="translate({translateX},{translateY})">
  {#if orientation === "left" || orientation == "right"}
    <g>
      {#each values as d}
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
            {@const anchor =
              textAlignToAnchor[
                getTickLabelTextAlign(orientation, tickLabelAngle)
              ]}
            {#if maxTickLabelSpace}
              <Label
                label={format(d)}
                width={maxTickLabelSpace}
                x={tickSpacing * k}
                y={0}
                dominantBaseline={"middle"}
                textAnchor={anchor}
                fontSize={tickLabelFontSize}
                fontColor={tickLabelColor}
                fontFamily={tickLabelFontFamily}
                angle={tickLabelAngle}
              />
            {:else}
              <text
                dominant-baseline={"middle"}
                text-anchor={anchor}
                transform="translate({tickSpacing *
                  k} 0) rotate({tickLabelAngle})"
                fill={tickLabelColor}
                font-size={tickLabelFontSize}
                font-family={tickLabelFontFamily}
              >
                {format(d)}
              </text>
            {/if}
          {/if}
        </g>
      {/each}
    </g>

    {#if showDomain}
      <line
        x1={0}
        x2={0}
        y1={scale.range()[0]}
        y2={scale.range()[1]}
        stroke-width={1}
        stroke={domainColor}
      />
    {/if}
  {:else}
    <g>
      {#each values as d}
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
            {@const anchor =
              textAlignToAnchor[
                getTickLabelTextAlign(orientation, tickLabelAngle)
              ]}
            {#if maxTickLabelSpace}
              <Label
                label={format(d)}
                width={maxTickLabelSpace}
                x={0}
                y={tickSpacing * k}
                dominantBaseline={orientation === "top"
                  ? "text-top"
                  : "hanging"}
                textAnchor={anchor}
                fontSize={tickLabelFontSize}
                fontFamily={tickLabelFontFamily}
                fontColor={tickLabelColor}
                angle={tickLabelAngle}
              />
            {:else}
              <text
                dominant-baseline={orientation === "top"
                  ? "text-top"
                  : "hanging"}
                text-anchor={anchor}
                transform="translate(0 {tickSpacing *
                  k}) rotate({tickLabelAngle})"
                font-size={tickLabelFontSize}
                font-family={tickLabelFontFamily}
                fill={tickLabelColor}>{format(d)}</text
              >
            {/if}
          {/if}
        </g>
      {/each}
    </g>

    {#if showDomain}
      <line
        x1={scale.range()[0]}
        x2={scale.range()[1]}
        y1={0}
        y2={0}
        stroke-width={1}
        stroke={domainColor}
      />
    {/if}
  {/if}

  {#if title}
    <text
      fill={titleColor}
      font-size={titleFontSize}
      font-family={titleFontFamily}
      font-weight={titleFontWeight}
      text-anchor={textAlignToAnchor[titleLocation.textAlign]}
      transform="translate({titleLocation.x} {titleLocation.y}) rotate({titleLocation.rotate}) translate({titleOffsetX} {titleOffsetY})"
    >
      {title}
    </text>
  {/if}
</g>
