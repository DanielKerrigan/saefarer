<!-- Axis component based on d3-axis -->

<script lang="ts" generics="Domain">
  import type { AxisScale } from "./axis";
  import Label from "./Label.svelte";

  let {
    orient,
    scale,
    translateX = 0,
    translateY = 0,
    tickLineSize = 6,
    tickLabelFontSize = 10,
    tickPadding = 3,
    tickFormat,
    numTicks,
    tickValues,
    showTickMarks = true,
    showTickLabels = true,
    tickLabelSpace,
    tickLineColor = "black",
    tickLabelColor = "black",
  }: {
    orient: "top" | "right" | "bottom" | "left";
    scale: AxisScale<Domain>;
    translateX?: number;
    translateY?: number;
    tickLineSize?: number;
    tickLabelFontSize?: number;
    tickPadding?: number;
    tickFormat?: (value: Domain) => string;
    numTicks?: number;
    tickValues?: Domain[];
    showTickMarks?: boolean;
    showTickLabels?: boolean;
    tickLabelSpace?: number;
    tickLineColor?: string;
    tickLabelColor?: string;
  } = $props();

  let k = $derived(orient === "top" || orient === "left" ? -1 : 1);
  let spacing = $derived(Math.max(tickLineSize, 0) + tickPadding);
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
</script>

<g transform="translate({translateX},{translateY})">
  {#each values as d}
    {#if orient === "left" || orient === "right"}
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
          {#if tickLabelSpace}
            <Label
              label={format(d)}
              width={tickLabelSpace}
              x={spacing * k}
              y={0}
              dominantBaseline={"middle"}
              textAnchor={orient === "left" ? "end" : "start"}
              fontSize={tickLabelFontSize}
              fontColor={tickLabelColor}
            />
          {:else}
            <text
              x={spacing * k}
              y={0}
              dominant-baseline={"middle"}
              text-anchor={orient === "left" ? "end" : "start"}
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
          {#if tickLabelSpace}
            <Label
              label={format(d)}
              width={tickLabelSpace}
              x={0}
              y={spacing * k}
              dominantBaseline={orient === "top" ? "text-top" : "hanging"}
              textAnchor={"middle"}
              fontSize={tickLabelFontSize}
              fontColor={tickLabelColor}
            />
          {:else}
            <text
              x={0}
              y={spacing * k}
              dominant-baseline={orient === "top" ? "text-top" : "hanging"}
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
