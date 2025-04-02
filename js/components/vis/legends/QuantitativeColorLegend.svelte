<!--
   References code from: https://observablehq.com/@d3/color-legend
   Copyright 2021, Observable Inc.
   Released under the ISC license.
 -->
<script lang="ts">
  import { scaleLinear } from "d3-scale";
  import type { ScaleSequential, ScaleDiverging } from "d3-scale";
  import { defaultFormat, scaleCanvas } from "../vis-utils";
  import { axis } from "../axis/axis";

  let {
    width,
    height,
    color,
    marginTop = 10,
    marginRight = 10,
    marginBottom = 10,
    marginLeft = 10,
    title = "",
  }: {
    width: number;
    height: number;
    color: ScaleSequential<string> | ScaleDiverging<string>;
    marginTop?: number;
    marginRight?: number;
    marginBottom?: number;
    marginLeft?: number;
    title?: string;
  } = $props();

  let canvas: HTMLCanvasElement | null = $state(null);
  let ctx: CanvasRenderingContext2D | null = $derived(
    canvas
      ? (canvas as HTMLCanvasElement).getContext("2d", {
          alpha: false,
        })
      : null,
  );

  // drawing

  function draw(
    ctx: CanvasRenderingContext2D,
    color: ScaleSequential<string> | ScaleDiverging<string>,
    width: number,
    height: number,
    marginTop: number,
    marginRight: number,
    marginBottom: number,
    marginLeft: number,
  ) {
    const x = scaleLinear()
      .domain([color.domain()[0], color.domain()[color.domain().length - 1]])
      .range([marginLeft, width - marginRight]);

    const colorWidth = x.range()[1] - x.range()[0];
    const colorHeight = height - marginTop - marginBottom;

    const minDesiredTicks = color.domain().length;
    const tickValues = x.ticks(
      Math.max(Math.min(colorWidth / 50, 10), minDesiredTicks),
    );

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, width, height);

    for (let i = 0; i < colorWidth; i++) {
      ctx.fillStyle = color.interpolator()(i / colorWidth);
      ctx.fillRect(i + marginLeft, marginTop, 1, colorHeight);
    }

    axis(ctx, "bottom", x, {
      translateY: height - marginBottom,
      tickValues,
      tickFormat: defaultFormat,
      title: title,
      titleAnchor: "left",
      titleOffsetX: marginLeft,
      titleOffsetY: -marginBottom - colorHeight,
      marginTop,
      marginRight,
      marginBottom,
      marginLeft,
    });
  }

  $effect(() => {
    if (canvas && ctx) {
      scaleCanvas(canvas, ctx, width, height);
    }
  });

  $effect(() => {
    if (ctx) {
      draw(
        ctx,
        color,
        width,
        height,
        marginTop,
        marginRight,
        marginBottom,
        marginLeft,
      );
    }
  });
</script>

<canvas bind:this={canvas}></canvas>
