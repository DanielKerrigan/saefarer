<script lang="ts">
  import type { Snippet } from "svelte";

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

  let width = $state(0);
  let height = $state(0);

  function getTop(height: number, rootRect: DOMRect, targetRect: DOMRect) {
    if (targetRect.top - height < rootRect.top) {
      // tooltip needs to go below target
      return targetRect.bottom - rootRect.top + space;
    } else {
      // tooltip defaults to above target
      return targetRect.top - rootRect.top - height - space;
    }
  }

  function getLeft(width: number, rootRect: DOMRect, targetRect: DOMRect) {
    const halfTooltipWidth = width / 2;
    const targetRectMiddle = (targetRect.left + targetRect.right) / 2;

    if (targetRectMiddle - halfTooltipWidth < rootRect.left) {
      // tooltip needs to the right
      return targetRect.right - rootRect.left + space;
    } else if (targetRectMiddle + halfTooltipWidth > rootRect.right) {
      // tooltip needs to the left
      return targetRect.left - rootRect.left - width - space;
    } else {
      // tooltip goes in center
      return (
        targetRect.left -
        rootRect.left +
        targetRect.width / 2 -
        halfTooltipWidth
      );
    }
  }

  let top = $derived(getTop(height, rootRect, targetRect));
  let left = $derived(getLeft(width, rootRect, targetRect));
</script>

<div
  class="sae-tooltip"
  bind:offsetWidth={width}
  bind:offsetHeight={height}
  style="left: {left}px; top: {top}px;"
>
  {@render content()}
</div>

<style>
  .sae-tooltip {
    padding: 0.5em;
    position: absolute;
    background-color: var(--color-white);
    border: 1px solid var(--color-black);
    color: var(--color-black);
    font-weight: normal;
    pointer-events: none;
    box-sizing: border-box;
    z-index: 10;
  }
</style>
