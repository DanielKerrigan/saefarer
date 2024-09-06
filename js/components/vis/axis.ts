import type { AxisDomain, AxisTimeInterval } from "d3-axis";
import type { ScaleLinear, ScaleBand, ScalePoint, ScaleTime } from "d3-scale";
import { defaultFormat, fitString } from "./vis-utils";

export type AxisScale<Domain> = {
  (x: Domain): number | undefined;
  domain(): Domain[];
  range(): number[];
  bandwidth?(): number;
  ticks?(count?: number): Domain[];
  tickFormat?(count?: number, specifier?: string): (d: Domain) => string;
};

export function axis<Domain extends number | string | Date>(
  orient: "top" | "right" | "bottom" | "left",
  ctx: CanvasRenderingContext2D,
  scale: AxisScale<Domain>,
  {
    translateX = 0,
    translateY = 0,
    tickSize = 6,
    tickPadding = 3,
    tickFormat,
    numTicks,
    tickValues,
    showTickMarks = true,
    showTickLabels = true,
    tickLabelSpace,
  }: {
    translateX?: number;
    translateY?: number;
    tickSize?: number;
    tickPadding?: number;
    tickFormat?: (value: Domain) => string;
    numTicks?: number;
    tickValues?: Domain[];
    showTickMarks?: boolean;
    showTickLabels?: boolean;
    tickLabelSpace?: number;
  } = {},
): void {
  const k = orient === "top" || orient === "left" ? -1 : 1;
  const spacing = Math.max(tickSize, 0) + tickPadding;
  const offset = scale.bandwidth ? scale.bandwidth() / 2 : 0;

  const values =
    tickValues ?? (scale.ticks ? scale.ticks(numTicks) : scale.domain());

  const format =
    tickFormat ??
    (scale.tickFormat
      ? scale.tickFormat(numTicks)
      : (d) => String(d).toString());

  ctx.save();

  ctx.translate(translateX, translateY);

  ctx.font = "10px sans-serif";
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "rgb(145, 145, 145)";
  ctx.fillStyle = "black";

  values.forEach((d) => {
    if (orient === "left" || orient === "right") {
      const y = (scale(d) ?? 0) + offset;

      if (showTickMarks) {
        ctx.beginPath();
        ctx.moveTo(tickSize * k, y);
        ctx.lineTo(0, y);
        ctx.stroke();
      }

      if (showTickLabels) {
        ctx.textBaseline = "middle";
        ctx.textAlign = orient === "left" ? "end" : "start";
        const tickLabel = tickLabelSpace
          ? fitString(ctx, format(d), tickLabelSpace)
          : format(d);
        ctx.fillText(tickLabel, spacing * k, y);
      }
    } else {
      const x = (scale(d) ?? 0) + offset;

      if (showTickMarks) {
        ctx.beginPath();
        ctx.moveTo(x, tickSize * k);
        ctx.lineTo(x, 0);
        ctx.stroke();
      }

      if (showTickLabels) {
        ctx.textBaseline = orient === "top" ? "bottom" : "top";
        ctx.textAlign = "center";
        const tickLabel = tickLabelSpace
          ? fitString(ctx, format(d), tickLabelSpace)
          : format(d);
        ctx.fillText(tickLabel, x, spacing * k);
      }
    }
  });

  ctx.restore();
}
