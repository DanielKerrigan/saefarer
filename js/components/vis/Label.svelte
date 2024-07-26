<script lang="ts">
  let {
    label,
    width,
    x,
    y,
    bold = false,
    rotate = false,
    fontSize = 10,
    fontColor = "black",
    dominantBaseline = "auto",
    textAnchor = "start",
  }: {
    label: string;
    width: number;
    x: number;
    y: number;
    bold?: boolean;
    rotate?: boolean;
    fontSize?: number;
    fontColor?: string;
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
  {x}
  {y}
  fill={fontColor}
  class="tw-select-none"
  class:tw-font-bold={bold}
  font-size={fontSize}
  transform={rotate ? `rotate(270, ${x}, ${y})` : null}
>
  <tspan
    dominant-baseline={dominantBaseline}
    text-anchor={textAnchor}
    bind:this={tspan}
  />
  <title>{label}</title>
</text>
