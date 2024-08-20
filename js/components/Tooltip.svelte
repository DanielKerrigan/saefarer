<script lang="ts">
  import type { TokenSequence } from "../types";
  import { format } from "d3-format";

  let {
    sequence,
    index,
  }: {
    sequence: TokenSequence;
    index: number;
  } = $props();

  const actFormat = format(".3f");

  let borderBoxSize: ResizeObserverSize[] | undefined = $state();

  const width = $derived(borderBoxSize ? borderBoxSize[0].inlineSize : 100);
  const height = $derived(borderBoxSize ? borderBoxSize[0].blockSize : 100);
</script>

<div
  class="sae-tooltip"
  bind:borderBoxSize
  style="left: {-width / 2}px; top: 2em;"
>
  <table class="sae-tooltip-table">
    <tbody>
      <tr>
        <td class="sae-string">Token:</td>
        <td class="sae-string">{sequence.token[index]}</td>
      </tr>
      <tr>
        <td class="sae-string">Activation:</td>
        <td class="sae-number">{actFormat(sequence.activation[index])}</td>
      </tr>
      {#each Object.entries(sequence.extras) as [name, values]}
        <tr>
          <td class="sae-string">{name}:</td>
          <td class="sae-string">{values[index]}</td>
        </tr>
      {/each}
    </tbody>
  </table>
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

  .sae-tooltip-table {
    border-collapse: collapse;
  }

  .sae-tooltip-table td {
    padding: 0em 0.5em 0.25em 0em;
    line-height: 1;
    vertical-align: middle;
  }

  /* no right padding for last column in table */
  .sae-tooltip-table tr > td:last-of-type {
    padding-right: 0;
  }

  /* no bottom padding for last row in table */
  .sae-tooltip-table tbody > tr:last-of-type > td {
    padding-bottom: 0;
  }

  .sae-tooltip-table td.sae-number {
    font-variant-numeric: lining-nums tabular-nums;
    text-align: right;
  }

  .sae-tooltip-table .sae-string {
    text-align: left;
  }
</style>
