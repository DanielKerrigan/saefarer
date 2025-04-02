/* Axis component inspired by d3-axis and Observable Plot */

// types

export type Orientation = "top" | "right" | "bottom" | "left";
export type TitleAnchor = "top" | "right" | "bottom" | "left" | "center";

export type Domain = number | string | Date;

export type Scale<D extends Domain> = {
  (x: D): number | undefined;
  domain(): D[];
  range(): number[];
  bandwidth?(): number;
  ticks?(count?: number): D[];
  tickFormat?(count?: number, specifier?: string): (d: D) => string;
};

// Canvas text cutoff

// https://stackoverflow.com/a/68395616
function binarySearch(
  maxIndex: number,
  getValue: (guess: number) => number,
  targetValue: number,
) {
  let minIndex = 0;

  while (minIndex <= maxIndex) {
    const guessIndex = Math.floor((minIndex + maxIndex) / 2);
    const guessValue = getValue(guessIndex);

    if (guessValue === targetValue) {
      return guessIndex;
    } else if (guessValue < targetValue) {
      minIndex = guessIndex + 1;
    } else {
      maxIndex = guessIndex - 1;
    }
  }

  return maxIndex;
}

// https://stackoverflow.com/a/68395616
function fitString(
  ctx: CanvasRenderingContext2D,
  str: string,
  maxWidth: number,
) {
  const width = ctx.measureText(str).width;
  const ellipsis = "…";
  const ellipsisWidth = ctx.measureText(ellipsis).width;

  if (width <= maxWidth || width <= ellipsisWidth) {
    return str;
  }

  const index = binarySearch(
    str.length - 1,
    (guess) => ctx.measureText(str.substring(0, guess)).width,
    maxWidth - ellipsisWidth,
  );

  return str.substring(0, index) + ellipsis;
}

// axis title

export function getTitleLocation<D extends Domain>(
  orientation: Orientation,
  titleAnchor: TitleAnchor,
  scale: Scale<D>,
  marginLeft: number,
  marginTop: number,
  marginRight: number,
  marginBottom: number,
  fontSize: number,
): {
  textAlign: "start" | "end" | "center";
  x: number;
  y: number;
  rotate: number;
} {
  const minRange = Math.min(...scale.range());
  const maxRange = Math.max(...scale.range());
  const midRange = (minRange + maxRange) / 2;

  const isLeft = orientation === "left";
  const isRight = orientation === "right";
  const isTop = orientation === "top";

  if (isLeft || isRight) {
    const x = isLeft ? -marginLeft : marginRight;
    if (titleAnchor === "top") {
      const yPos = minRange - marginTop;
      const dy = 0.71 * fontSize;
      return {
        textAlign: isLeft ? "start" : "end",
        x,
        y: yPos + dy,
        rotate: 0,
      };
    } else if (titleAnchor === "bottom") {
      return {
        textAlign: isLeft ? "start" : "end",
        x,
        y: maxRange + marginBottom,
        rotate: 0,
      };
    } else {
      const k = isLeft ? 1 : -1;
      const xPos = x + (k * fontSize) / 2;
      const yPos = midRange;
      const dy = k * 0.71 * fontSize;
      return {
        textAlign: "center",
        x: xPos + dy,
        y: yPos,
        rotate: isLeft ? -90 : 90,
      };
    }
  } else {
    const y = isTop ? -marginTop + fontSize / 2 : marginBottom - fontSize / 2;
    const dy = isTop ? fontSize * 0.71 : 0;
    if (titleAnchor === "left") {
      return {
        textAlign: "start",
        x: minRange - marginLeft,
        y: y + dy,
        rotate: 0,
      };
    } else if (titleAnchor === "right") {
      return {
        textAlign: "end",
        x: maxRange + marginRight,
        y: y + dy,
        rotate: 0,
      };
    } else {
      return {
        textAlign: "center",
        x: midRange,
        y: y + dy,
        rotate: 0,
      };
    }
  }
}

// text anchor

export function getTickLabelTextAlign(
  orientation: Orientation,
  angle: number,
): "start" | "center" | "end" {
  if (orientation === "left") {
    return "end";
  } else if (orientation === "right") {
    return "start";
  } else if (angle === 0) {
    return "center";
  } else if (angle > 0 && orientation === "top") {
    return "end";
  } else if (angle < 0 && orientation === "top") {
    return "start";
  } else if (angle > 0 && orientation === "bottom") {
    return "start";
  } else {
    return "end";
  }
}

export const textAlignToAnchor = {
  start: "start" as const,
  center: "middle" as const,
  end: "end" as const,
};

// canvas drawing

export function axis<D extends Domain>(
  ctx: CanvasRenderingContext2D,
  orientation: Orientation,
  scale: Scale<D>,
  {
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
    titleAnchor?: TitleAnchor;
    titleOffsetX?: number;
    titleOffsetY?: number;
    titleColor?: string;
  } = {},
): void {
  const k = orientation === "top" || orientation === "left" ? -1 : 1;
  const tickSpacing = Math.max(tickLineSize, 0) + tickPadding;
  const offset = scale.bandwidth ? scale.bandwidth() / 2 : 0;

  const titleLocation = getTitleLocation(
    orientation,
    titleAnchor,
    scale,
    marginLeft,
    marginTop,
    marginRight,
    marginBottom,
    titleFontSize,
  );

  const values =
    tickValues ?? (scale.ticks ? scale.ticks(numTicks) : scale.domain());

  const format =
    tickFormat ??
    (scale.tickFormat
      ? scale.tickFormat(numTicks)
      : (d: D) => String(d).toString());

  ctx.save();

  ctx.translate(translateX, translateY);

  ctx.font = `${tickLabelFontSize}px ${tickLabelFontFamily}`;
  ctx.globalAlpha = 1;

  ctx.fillStyle = tickLabelColor;
  ctx.strokeStyle = tickLineColor;

  if (orientation === "left" || orientation === "right") {
    values.forEach((d) => {
      const y = (scale(d) ?? 0) + offset;

      if (showTickMarks) {
        ctx.beginPath();
        ctx.moveTo(tickLineSize * k, y);
        ctx.lineTo(0, y);
        ctx.stroke();
      }

      if (showTickLabels) {
        ctx.save();
        ctx.translate(tickSpacing * k, y);
        ctx.rotate((tickLabelAngle * Math.PI) / 2);
        ctx.textBaseline = "middle";
        ctx.textAlign = orientation === "left" ? "end" : "start";
        const tickLabel = maxTickLabelSpace
          ? fitString(ctx, format(d), maxTickLabelSpace)
          : format(d);
        ctx.fillText(tickLabel, 0, 0);
        ctx.restore();
      }
    });

    ctx.strokeStyle = domainColor;
    ctx.lineWidth = 1;

    if (showDomain) {
      ctx.beginPath();
      ctx.moveTo(0, scale.range()[0]);
      ctx.lineTo(0, scale.range()[1]);
      ctx.stroke();
    }
  } else {
    values.forEach((d) => {
      const x = (scale(d) ?? 0) + offset;

      if (showTickMarks) {
        ctx.beginPath();
        ctx.moveTo(x, tickLineSize * k);
        ctx.lineTo(x, 0);
        ctx.stroke();
      }

      if (showTickLabels) {
        ctx.save();
        ctx.translate(x, tickSpacing * k);
        ctx.rotate((tickLabelAngle * Math.PI) / 2);
        ctx.textBaseline = orientation === "top" ? "bottom" : "top";
        ctx.textAlign = "center";
        const tickLabel = maxTickLabelSpace
          ? fitString(ctx, format(d), maxTickLabelSpace)
          : format(d);
        ctx.fillText(tickLabel, 0, 0);
        ctx.restore();
      }
    });

    ctx.strokeStyle = domainColor;
    ctx.lineWidth = 1;

    if (showDomain) {
      ctx.beginPath();
      ctx.moveTo(scale.range()[0], 0);
      ctx.lineTo(scale.range()[1], 0);
      ctx.stroke();
    }
  }

  if (title) {
    ctx.fillStyle = titleColor;
    ctx.textAlign = titleLocation.textAlign;
    ctx.textBaseline = "alphabetic";
    ctx.font = `${titleFontWeight} ${titleFontSize}px ${titleFontFamily}`;
    ctx.translate(titleLocation.x, titleLocation.y);
    ctx.rotate((titleLocation.rotate * Math.PI) / 2);
    ctx.fillText(title, titleOffsetX, titleOffsetY);
  }

  ctx.restore();
}
