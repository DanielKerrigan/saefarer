<script lang="ts">
  import type { ScaleSequential } from "d3-scale";
  import type { FeatureTokenSequence, DisplayToken } from "../types";
  import VisTooltip from "./VisTooltip.svelte";
  import { activationValueFormat } from "./vis/vis-utils";
  import TooltipTable from "./TooltipTable.svelte";

  let {
    colorScale,
    sequence,
    wrap,
    tooltipEnabled = true,
    hidePadding = false,
  }: {
    colorScale: ScaleSequential<string>;
    sequence: FeatureTokenSequence;
    wrap: boolean;
    tooltipEnabled?: boolean;
    hidePadding?: boolean;
  } = $props();

  let tooltipInfo: {
    data: DisplayToken;
    anchor: HTMLElement;
    index: number;
  } | null = $state(null);

  function onMouseEnterToken(
    event: MouseEvent & {
      currentTarget: EventTarget & HTMLDivElement;
    },
    data: DisplayToken,
    index: number,
  ) {
    tooltipInfo = {
      data,
      anchor: event.currentTarget,
      index,
    };
  }

  function onMouseLeaveToken() {
    tooltipInfo = null;
  }
</script>

<div class="sae-sequence" style:flex-wrap={wrap ? "wrap" : "nowrap"}>
  {#each sequence.display_tokens as dt, i}
    {#if !(hidePadding && dt.is_padding)}
      {@const tokenColor =
        dt.max_act > 0 ? colorScale(dt.max_act) : "var(--color-white)"}
      <!-- TODO: do this properly -->
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <div
        class="sae-token"
        style:--token-color={tokenColor}
        style:font-weight={i === sequence.max_token_index && dt.max_act > 0
          ? "var(--font-bold)"
          : "var(--font-normal)"}
        style:background-color={tooltipEnabled && tooltipInfo?.index === i
          ? "var(--color-neutral-300)"
          : "var(--color-white)"}
        onmouseenter={tooltipEnabled
          ? (event) => onMouseEnterToken(event, dt, i)
          : null}
        onmouseleave={tooltipEnabled ? onMouseLeaveToken : null}
      >
        <span class="sae-token-name">{dt.display}</span>
      </div>
    {/if}
  {/each}

  {#if tooltipInfo}
    <VisTooltip {...tooltipInfo}>
      {#if tooltipInfo}
        <TooltipTable
          data={[
            { key: "Token", value: tooltipInfo.data.display },
            {
              key: "Activation",
              value: activationValueFormat(tooltipInfo.data.max_act),
            },
          ]}
        />
      {/if}
    </VisTooltip>
  {/if}
</div>

<style>
  .sae-sequence {
    display: flex;
  }

  .sae-token {
    line-height: 1.2;
    border-bottom: 0.25em solid var(--token-color);
    background-color: var(--color-white);
    color: var(--color-black);
  }

  .sae-token-name {
    white-space: pre;
  }
</style>
