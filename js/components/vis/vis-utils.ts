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

export function defaultFormat(x: number): string {
  /* [0, 1] is a common range for predictions and features.
    With SI suffixes, 0.5 becomes 500m. I'd rather it just be 0.5. */

  if ((x >= 0.001 && x <= 1) || (x >= -1 && x <= 0.001)) {
    return format(".3~f")(x);
  } else {
    return format("~s")(x);
  }
}

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
export function fitString(
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
