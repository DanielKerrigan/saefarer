<script lang="ts">
  let {
    label,
    width,
    x,
    y,
    angle = 0,
    fontSize = 10,
    fontFamily = "ui-sans-serif, system-ui, sans-serif",
    fontColor = "black",
    fontWeight = 400,
    dominantBaseline = "auto",
    textAnchor = "start",
  }: {
    label: string;
    width: number;
    x: number;
    y: number;
    bold?: boolean;
    angle?: number;
    fontSize?: number;
    fontFamily?: string;
    fontColor?: string;
    fontWeight?: number;
    dominantBaseline?:
      | "auto"
      | "text-bottom"
      | "alphabetic"
      | "ideographic"
      | "middle"
      | "central"
      | "mathematical"
      | "hanging"
      | "text-top";
    textAnchor?: "start" | "middle" | "end";
  } = $props();

  let tspan: SVGTSpanElement;

  function updateText(label: string, width: number) {
    if (!tspan) {
      return;
    }

    tspan.textContent = label;

    let part = label;

    while (part.length > 0 && tspan.getComputedTextLength() > width) {
      part = part.slice(0, -1);
      tspan.textContent = part + "…";
    }
  }

  $effect(() => {
    updateText(label, width);
  });
</script>

<text
  fill={fontColor}
  font-size={fontSize}
  font-family={fontFamily}
  font-weight={fontWeight}
  transform="translate({x} {y}) rotate({angle})"
>
  <tspan
    dominant-baseline={dominantBaseline}
    text-anchor={textAnchor}
    bind:this={tspan}
  />
  <title>{label}</title>
</text>
