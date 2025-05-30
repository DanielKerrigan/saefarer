<script lang="ts">
  let {
    width,
    height,
    x,
    y,
    color1 = "var(--color-black)",
    color2 = "var(--color-white)",
    strokeWidth = 1,
    strokeDashArray = "4",
  }: {
    width: number;
    height: number;
    x: number;
    y: number;
    color1?: string;
    color2?: string;
    strokeWidth?: number;
    strokeDashArray?: string;
  } = $props();
</script>

<g transform="translate({x},{y})">
  <rect
    {width}
    {height}
    stroke-width={strokeWidth}
    stroke={color1}
    fill="none"
  />
  <rect
    {width}
    {height}
    stroke-width={strokeWidth}
    stroke={color2}
    stroke-dasharray={strokeDashArray}
    fill="none"
  />
</g>
