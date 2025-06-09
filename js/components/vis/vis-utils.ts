import { format } from "d3-format";

// Adapted from https://www.html5rocks.com/en/tutorials/canvas/hidpi/
export function scaleCanvas(
  canvas: HTMLCanvasElement,
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
): void {
  // assume the device pixel ratio is 1 if the browser doesn't specify it
  const devicePixelRatio = window.devicePixelRatio || 1;

  // set the 'real' canvas size to the higher width/height
  canvas.width = width * devicePixelRatio;
  canvas.height = height * devicePixelRatio;

  // ...then scale it back down with CSS
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;

  // scale the drawing context so everything will work at the higher ratio
  context.scale(devicePixelRatio, devicePixelRatio);
}

export function getSizeWithAspectRatio(
  maxWidth: number,
  maxHeight: number,
  aspectRatio: number,
): { width: number; height: number } {
  const unit = Math.min(maxWidth / aspectRatio, maxHeight);

  return {
    width: aspectRatio * unit,
    height: unit,
  };
}

export function getSizeWithAspectRatioMargins(
  maxWidth: number,
  maxHeight: number,
  aspectRatio: number,
  marginTop: number,
  marginRight: number,
  marginBottom: number,
  marginLeft: number,
): { width: number; height: number } {
  const maxW = maxWidth - marginRight - marginLeft;
  const maxH = maxHeight - marginTop - marginBottom;
  const size = getSizeWithAspectRatio(maxW, maxH, aspectRatio);
  return {
    width: size.width + marginRight + marginLeft,
    height: size.height + marginTop + marginBottom,
  };
}

export function defaultFormat(x: number): string {
  /* [0, 1] is a common range for predictions and features.
    With SI suffixes, 0.5 becomes 500m. I'd rather it just be 0.5. */

  if ((x >= 0.001 && x <= 1) || (x >= -1 && x <= 0.001)) {
    return format(".3~f")(x);
  } else {
    return format("~s")(x);
  }
}

export function activationRatePctFormat(x: number): string {
  if (x < 0.00001) {
    return format(".1~p")(x);
  } else if (x < 0.001) {
    return format(".2~p")(x);
  } else {
    return format(".3~p")(x);
  }
}

export const activationRateLogFormat = format(".3~f");
export const activationValueFormat = format(".2~f");
export const probabilityFormat = format(".2~f");
export const logLossFormat = format(".3~f");
export const percentFormat = format(".2~%");
export const percentagePointFormat = (d: number) => format(".2~f")(d * 100);
export const countFormat = format(",d");
export const siFormat = format(".3~s");

export const actValueHistogramTooltipData = [
  {
    key: "Instance count",
    value: (_x1: number, _x2: number, y: number) => countFormat(y),
  },
  {
    key: "Activation value",
    value: (x1: number, x2: number, _y: number) =>
      `${activationValueFormat(x1)} to ${activationValueFormat(x2)}`,
  },
];
