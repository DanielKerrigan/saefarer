<script lang="ts">
  import { hcl } from "d3-color";
  import type { ScaleSequential } from "d3-scale";
  import type { FeatureTokenSequence, DisplayToken } from "../types";
  import { rootDiv } from "../state.svelte";
  import TokenTooltipContent from "./TokenTooltipContent.svelte";
  import Tooltip from "./Tooltip.svelte";

  let {
    color,
    sequence,
    wrap,
  }: {
    color: ScaleSequential<string>;
    sequence: FeatureTokenSequence;
    wrap: boolean;
  } = $props();

  let tooltipInfo: {
    data: DisplayToken;
    rootRect: DOMRect;
    targetRect: DOMRect;
  } | null = $state(null);

  function onMouseEnterToken(
    event: MouseEvent & {
      currentTarget: EventTarget & HTMLDivElement;
    },
    data: DisplayToken,
  ) {
    if (!rootDiv.value) {
      return;
    }

    const targetRect = event.currentTarget.getBoundingClientRect();
    const rootRect = rootDiv.value.getBoundingClientRect();

    tooltipInfo = {
      data,
      rootRect,
      targetRect,
    };
  }

  function onMouseLeaveToken() {
    tooltipInfo = null;
  }
</script>

<div class="sae-sequence" style:flex-wrap={wrap ? "wrap" : "nowrap"}>
  {#each sequence.display_tokens as dt, i}
    {@const tokenColor = color(dt.max_act)}
    <!-- TODO: do this properly -->
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div
      class="sae-token"
      style:--border-color={tokenColor}
      style:background={tokenColor}
      style:color={hcl(tokenColor).l > 50
        ? "var(--color-black)"
        : "var(--color-white)"}
      style:font-weight={i === sequence.max_token_index ? "bold" : "normal"}
      onmouseenter={(event) => onMouseEnterToken(event, dt)}
      onmouseleave={onMouseLeaveToken}
    >
      <span class="sae-token-name">{dt.display}</span>
    </div>
  {/each}

  {#if tooltipInfo}
    <Tooltip {...tooltipInfo}>
      {#snippet content()}
        {#if tooltipInfo}
          <TokenTooltipContent data={tooltipInfo.data} />
        {/if}
      {/snippet}
    </Tooltip>
  {/if}
</div>

<style>
  .sae-sequence {
    display: flex;
  }

  .sae-token {
    border: 1px solid var(--border-color);
  }

  .sae-token:hover {
    border-color: var(--color-red-600);
  }

  .sae-token-name {
    white-space: pre;
  }
</style>
