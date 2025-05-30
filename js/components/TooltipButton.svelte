<script lang="ts">
  import type { Snippet } from "svelte";
  import { root } from "../state.svelte";

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

  // dimensions and location

  function getTop(
    contentHeight: number,
    rootRect: DOMRect,
    anchorRect: DOMRect,
    space: number,
    position: Position,
  ) {
    const halfContentHeight = contentHeight / 2;
    const halfAnchorHeight = anchorRect.height / 2;

    if (position === "right" || position === "left") {
      return (
        anchorRect.top - rootRect.top + halfAnchorHeight - halfContentHeight
      );
    } else if (
      position === "bottom" ||
      (position === "auto" && anchorRect.top - contentHeight < rootRect.top)
    ) {
      // below
      return anchorRect.bottom - rootRect.top + space;
    } else {
      // above
      return anchorRect.top - rootRect.top - contentHeight - space;
    }
  }

  function getLeft(
    contentWidth: number,
    rootRect: DOMRect,
    anchorRect: DOMRect,
    space: number,
    position: Position,
  ) {
    const halfContentWidth = contentWidth / 2;
    const anchorRectMiddle =
      anchorRect.left - rootRect.left + anchorRect.width / 2;

    if (
      position === "right" ||
      (position === "auto" &&
        anchorRectMiddle - halfContentWidth < rootRect.left)
    ) {
      // right
      return anchorRect.right - rootRect.left + space;
    } else if (
      position === "left" ||
      (position === "auto" &&
        anchorRectMiddle + halfContentWidth > rootRect.right)
    ) {
      // left
      return anchorRect.left - rootRect.left - contentWidth - space;
    } else {
      // center
      return anchorRectMiddle - halfContentWidth;
    }
  }

  const space = 4;

  let anchor: HTMLButtonElement | undefined = $state();

  let contentWidth = $state(0);
  let contentHeight = $state(0);

  const anchorRect = $derived(
    anchor === undefined
      ? new DOMRect(0, 0, 0, 0)
      : anchor.getBoundingClientRect(),
  );
  const rootRect = $derived(root.value.getBoundingClientRect());

  let top = $derived(
    getTop(contentHeight, rootRect, anchorRect, space, position),
  );
  let left = $derived(
    getLeft(contentWidth, rootRect, anchorRect, space, position),
  );

  // opening and closing

  let show = $state(false);
  let locked = $state(false);

  function onclick() {
    locked = !locked;
    show = locked;
  }

  function onmouseenter() {
    if (!locked) {
      show = true;
    }
  }

  function onmouseleave() {
    if (!locked) {
      show = false;
    }
  }
</script>

<div class="sae-tooltip-container">
  <button {onclick} {onmouseenter} {onmouseleave} bind:this={anchor}>
    {@render trigger()}
  </button>

  {#if show}
    <div
      class="sae-tooltip-content"
      bind:offsetWidth={contentWidth}
      bind:offsetHeight={contentHeight}
      style:top="{top}px"
      style:left="{left}px"
    >
      {@render content()}
    </div>
  {/if}
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

  .sae-tooltip-content {
    padding: 0.5em;
    position: fixed;
    background-color: var(--color-white);
    border: 1px solid var(--color-black);
    color: var(--color-black);
    font-weight: var(--font-normal);
    pointer-events: none;
    z-index: 10;
  }
</style>
