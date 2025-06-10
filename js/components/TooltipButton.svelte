<script lang="ts">
  import type { Snippet } from "svelte";
  import { rootDiv } from "../state.svelte";

  type Position = "top" | "right" | "bottom" | "left" | "auto";

  let {
    position = "auto",
    trigger,
    content,
  }: {
    position?: Position;
    trigger: Snippet;
    content: Snippet;
  } = $props();

  const uid = $props.id();

  // dimensions and location

  function getTop(
    contentHeight: number,
    rootRect: DOMRect | null,
    anchorRect: DOMRect | null,
    space: number,
    position: Position,
  ) {
    if (rootRect === null || anchorRect === null) {
      return 0;
    }

    const halfContentHeight = contentHeight / 2;
    const halfAnchorHeight = anchorRect.height / 2;

    if (position === "right" || position === "left") {
      return anchorRect.top + halfAnchorHeight - halfContentHeight;
    } else if (
      position === "bottom" ||
      (position === "auto" && anchorRect.top - contentHeight < rootRect.top)
    ) {
      // below
      return anchorRect.bottom + space;
    } else {
      // above
      return anchorRect.top - contentHeight - space;
    }
  }

  function getLeft(
    contentWidth: number,
    rootRect: DOMRect | null,
    anchorRect: DOMRect | null,
    space: number,
    position: Position,
  ) {
    if (rootRect === null || anchorRect === null) {
      return 0;
    }

    const halfContentWidth = contentWidth / 2;
    const anchorRectMiddle = anchorRect.left + anchorRect.width / 2;

    if (
      position === "right" ||
      (position === "auto" &&
        anchorRectMiddle - halfContentWidth < rootRect.left)
    ) {
      // right
      return anchorRect.right + space;
    } else if (
      position === "left" ||
      (position === "auto" &&
        anchorRectMiddle + halfContentWidth > rootRect.right)
    ) {
      // left
      return anchorRect.left - contentWidth - space;
    } else {
      // center
      return anchorRectMiddle - halfContentWidth;
    }
  }

  const space = 4;

  let anchor: HTMLButtonElement | undefined = $state();
  let tooltip: HTMLDivElement | undefined = $state();

  let tooltipWidth = $state(0);
  let tooltipHeight = $state(0);

  let anchorRect: DOMRect | null = $state(null);
  let rootRect: DOMRect | null = $state(null);

  let top = $derived(
    getTop(tooltipHeight, rootRect, anchorRect, space, position),
  );
  let left = $derived(
    getLeft(tooltipWidth, rootRect, anchorRect, space, position),
  );

  // opening and closing

  function onclick(event: MouseEvent) {
    event.preventDefault();
  }

  function onmouseenter() {
    if (anchor && tooltip) {
      anchorRect = anchor.getBoundingClientRect();
      rootRect = rootDiv.value.getBoundingClientRect();
      tooltip.showPopover();
    }
  }

  function onmouseleave() {
    if (anchor && tooltip) {
      tooltip.hidePopover();
    }
  }
</script>

<div class="sae-tooltip-container">
  <button
    bind:this={anchor}
    popovertarget={uid}
    {onclick}
    {onmouseenter}
    {onmouseleave}
  >
    {@render trigger()}
  </button>

  <div
    bind:this={tooltip}
    id={uid}
    popover="auto"
    bind:offsetWidth={tooltipWidth}
    bind:offsetHeight={tooltipHeight}
    style:top="{top}px"
    style:left="{left}px"
  >
    {@render content()}
  </div>
</div>

<style>
  button,
  button:hover,
  button:active {
    border: none;
  }

  .sae-tooltip-container {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  [popover] {
    inset: unset;
    padding: 0.5em;
    position: fixed;
    background-color: var(--color-white);
    border: 1px solid var(--color-black);
    color: var(--color-black);
    font-weight: var(--font-normal);
    pointer-events: none;
  }
</style>
