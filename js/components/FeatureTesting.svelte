<script lang="ts">
  import type { ScaleSequential } from "d3-scale";
  import TokenSequence from "./TokenSequence.svelte";
  import {
    dataset_info,
    font_sizes,
    inference_input,
    inference_output,
  } from "../synced-state.svelte";
  import TooltipButton from "./TooltipButton.svelte";
  import TooltipTable from "./TooltipTable.svelte";
  import { percentFormat } from "./vis/vis-utils";
  import InfoIcon from "./icons/InfoIcon.svelte";

  let {
    tokenColor,
    featureId,
  }: {
    tokenColor: ScaleSequential<string>;
    featureId: number;
  } = $props();

  let wrapSequence = $state(false);
  let hidePadding = $state(true);

  let inferenceSequence = $derived(
    featureId === inference_input.value.feature_index
      ? inference_input.value.sequence
      : "",
  );

  function onTestFeature() {
    inference_input.value = {
      feature_index: featureId,
      sequence: inferenceSequence,
    };
  }

  function onkeydown(event: KeyboardEvent) {
    if (event.key === "Enter") {
      onTestFeature();
    }
  }
</script>

<div class="sae-feature-testing-container">
  <div class="sae-controls">
    <div class="sae-title">
      <span>Test Feature</span>
      <TooltipButton position="right">
        {#snippet trigger()}
          <InfoIcon />
        {/snippet}
        {#snippet content()}
          <div class="sae-info">
            Enter some text and check to see if it causes the feature to
            activate. Special tokens are automatically added.
          </div>
        {/snippet}
      </TooltipButton>
    </div>
    <label>
      <input type="checkbox" bind:checked={hidePadding} />
      <span>Hide padding</span>
    </label>
    <label>
      <input type="checkbox" bind:checked={wrapSequence} />
      <span>Wrap text</span>
    </label>
  </div>

  <div class="sae-input-row">
    <input type="text" bind:value={inferenceSequence} {onkeydown} />
    <button onclick={onTestFeature}>Test</button>
  </div>

  {#if inference_output.value.feature_index === featureId}
    <div class="sae-sequences-table">
      <div class="sae-sequences-table-cell sae-sequences-table-header"></div>
      <div class="sae-sequences-table-cell sae-sequences-table-header">
        Pred.
      </div>
      <div class="sae-sequences-table-cell sae-sequences-table-header">
        Prob.
      </div>
      <div class="sae-sequences-table-cell sae-sequences-table-header">
        Tokens
      </div>
      <div
        class="sae-sequences-table-cell"
        class:sae-sequences-table-border={true}
      >
        <TooltipButton position="left">
          {#snippet trigger()}
            <svg
              width="{font_sizes.base}px"
              height="{font_sizes.base}px"
              stroke-width="2"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              color="currentcolor"
            >
              <path
                d="M12 11.5V16.5"
                stroke="currentcolor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
              <path
                d="M12 7.51L12.01 7.49889"
                stroke="currentcolor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
              <path
                d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z"
                stroke="currentcolor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
          {/snippet}

          {#snippet content()}
            <TooltipTable data={[{ key: "Instance index", value: `-1` }]} />
          {/snippet}
        </TooltipButton>
      </div>
      <div
        class="sae-sequences-table-cell"
        class:sae-sequences-table-border={true}
      >
        {dataset_info.value.labels[inference_output.value.pred_label]}
      </div>
      <div
        class="sae-sequences-table-cell"
        class:sae-sequences-table-border={true}
      >
        {percentFormat(
          inference_output.value.pred_probs[inference_output.value.pred_label],
        )}
      </div>
      <div
        class="sae-sequences-table-cell sae-sequences-table-tokens"
        class:sae-sequences-table-border={true}
      >
        <TokenSequence
          colorScale={tokenColor}
          sequence={inference_output.value}
          wrap={wrapSequence}
          {hidePadding}
        />
      </div>
    </div>
  {/if}
</div>

<style>
  label {
    display: flex;
    align-items: center;
    gap: 0.25em;
  }

  .sae-controls {
    display: flex;
    gap: 1em;
    align-items: center;
    justify-content: flex-start;
  }

  .sae-feature-testing-container {
    min-height: 0;
    max-height: 100%;
    display: flex;
    flex-direction: column;
    gap: 0.5em;
  }

  .sae-input-row {
    display: flex;
    gap: 0.25em;
  }

  .sae-input-row input {
    border: 1px solid var(--color-black);
    border-radius: 0.25em;
    padding: 0em 0.25em;
    width: 100%;
  }

  .sae-sequences-table {
    min-height: 0;
    overflow-y: auto;
    display: grid;
    grid-auto-rows: max-content;
    grid-template-columns:
      max-content
      max-content
      max-content
      minmax(0, 1fr);
    border-top: 1px solid var(--color-neutral-500);
    border-bottom: 1px solid var(--color-neutral-500);
  }

  .sae-sequences-table-cell {
    padding: 0.25em 0.5em;
    display: flex;
    align-items: center;
  }

  .sae-sequences-table-header {
    text-transform: uppercase;
    font-weight: var(--font-medium);
    position: sticky;
    top: 0;
    z-index: 10;
    background-color: var(--color-white);
  }

  .sae-sequences-table-border {
    border-bottom: 1px solid var(--color-neutral-300);
  }

  .sae-sequences-table-tokens {
    overflow-x: auto;
  }

  .sae-title {
    display: flex;
    gap: 0.25em;
    align-items: center;
  }

  .sae-info {
    font-size: var(--text-sm);
    font-size: var(--text-sm);
    max-width: 16em;
  }

  .sae-title > span {
    font-weight: var(--font-medium);
  }
</style>
