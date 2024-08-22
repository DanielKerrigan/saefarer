<script lang="ts">
  import type { Snippet } from "svelte";
  import { BorderBoxSize } from "../state.svelte";

  let {
    rootRect,
    targetRect,
    content,
  }: {
    rootRect: DOMRect;
    targetRect: DOMRect;
    content: Snippet;
  } = $props();

  const space = 4;

  const tooltipDimensions = new BorderBoxSize([]);

  function getTop(tooltipDimensions: BorderBoxSize, rootRect: DOMRect, targetRect: DOMRect) {
    if (targetRect.top - tooltipDimensions.height < rootRect.top) {
      // tooltip needs to go below target
      return targetRect.bottom - rootRect.top + space;
    } else {
      // tooltip defaults to above target
      return targetRect.top - rootRect.top - tooltipDimensions.height - space;
    }
  }

  function getLeft(tooltipDimensions: BorderBoxSize, rootRect: DOMRect, targetRect: DOMRect) {
    const halfTooltipWidth = tooltipDimensions.width / 2

    if (targetRect.left - halfTooltipWidth < rootRect.left) {
      // tooltip needs to the right
      return targetRect.right - rootRect.left + space;
    } else if (targetRect.right + halfTooltipWidth > rootRect.right) {
      // tooltip needs to the left
      return targetRect.left - rootRect.left - tooltipDimensions.width - space;
    } else {
      // tooltip goes in center
      return targetRect.left - rootRect.left + (targetRect.width / 2) - halfTooltipWidth;
    }
  }

  let top = $derived(getTop(tooltipDimensions, rootRect, targetRect));
  let left = $derived(getLeft(tooltipDimensions, rootRect, targetRect));
</script>

<div
  class="sae-tooltip"
  bind:borderBoxSize={tooltipDimensions.borderBoxSize}
  style="left: {left}px; top: {top}px;"
>
  {@render content()}
</div>

<style>
  .sae-tooltip {
    padding: 0.5em;
    position: absolute;
    background-color: white;
    border: 1px solid black;
    color: black;
    font-weight: normal;
    pointer-events: none;
    box-sizing: border-box;
    z-index: 10;
  }
</style>
