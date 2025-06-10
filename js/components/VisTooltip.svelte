<script lang="ts">
  import type { Snippet } from "svelte";
  import { rootDiv } from "../state.svelte";

  let {
    anchor,
    children,
  }: {
    anchor: Element;
    children: Snippet;
  } = $props();

  function getTop(
    height: number,
    rootRect: DOMRect,
    anchorRect: DOMRect,
    space: number,
  ) {
    if (anchorRect.top - height < rootRect.top) {
      // tooltip needs to go below target
      return anchorRect.bottom - rootRect.top + space;
    } else {
      // tooltip defaults to above target
      return anchorRect.top - rootRect.top - height - space;
    }
  }

  function getLeft(
    width: number,
    rootRect: DOMRect,
    anchorRect: DOMRect,
    space: number,
  ) {
    const halfTooltipWidth = width / 2;
    const anchorRectMiddle = (anchorRect.left + anchorRect.right) / 2;

    if (anchorRectMiddle - halfTooltipWidth < rootRect.left) {
      // tooltip needs to the right
      return anchorRect.right - rootRect.left + space;
    } else if (anchorRectMiddle + halfTooltipWidth > rootRect.right) {
      // tooltip needs to the left
      return anchorRect.left - rootRect.left - width - space;
    } else {
      // tooltip goes in center
      return (
        anchorRect.left -
        rootRect.left +
        anchorRect.width / 2 -
        halfTooltipWidth
      );
    }
  }

  const space = 4;

  let width = $state(0);
  let height = $state(0);

  const anchorRect = $derived(anchor.getBoundingClientRect());
  const rootRect = $derived(rootDiv.value.getBoundingClientRect());

  let top = $derived(getTop(height, rootRect, anchorRect, space));
  let left = $derived(getLeft(width, rootRect, anchorRect, space));
</script>

<div
  class="sae-tooltip"
  bind:offsetWidth={width}
  bind:offsetHeight={height}
  style="left: {left}px; top: {top}px;"
>
  {@render children()}
</div>

<style>
  .sae-tooltip {
    padding: 0.5em;
    position: absolute;
    background-color: var(--color-white);
    border: 1px solid var(--color-black);
    color: var(--color-black);
    font-weight: var(--font-normal);
    pointer-events: none;
    box-sizing: border-box;
    z-index: 10;
  }
</style>
