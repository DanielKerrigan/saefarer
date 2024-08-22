<script lang="ts">
  import { format } from "d3-format";
  import type { FeatureToken } from "../types";

  let { data }: { data: FeatureToken } = $props();

  const actFormat = format(".3f");
</script>

<div class="sae-tooltip-content-container">
  <table class="sae-tooltip-table">
    <tbody>
      <tr>
        <td class="sae-string">Token:</td>
        <td class="sae-string">{data.token}</td>
      </tr>
      <tr>
        <td class="sae-string">Activation:</td>
        <td class="sae-number">{actFormat(data.activation)}</td>
      </tr>
      {#each data.extras as { key, value }}
        <tr>
          <td class="sae-string">{key}:</td>
          <td class="sae-string">{value}</td>
        </tr>
      {/each}
    </tbody>
  </table>
</div>

<style>
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
