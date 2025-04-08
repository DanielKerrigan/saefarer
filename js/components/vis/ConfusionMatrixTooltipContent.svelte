<script lang="ts">
  import { format } from "d3-format";
  import { model_info } from "../../synced-state.svelte";
  import type { ConfusionMatrixCell } from "../../types";

  let { data }: { data: ConfusionMatrixCell } = $props();

  const pctFormat = format(".1%");
</script>

<div class="sae-tooltip-content-container">
  <table class="sae-tooltip-table">
    <tbody>
      <tr>
        <td class="sae-string">True label:</td>
        <td class="sae-string">{model_info.value.labels[data.label]}</td>
      </tr>
      <tr>
        <td class="sae-string">Predicted label:</td>
        <td class="sae-string">{model_info.value.labels[data.pred_label]}</td>
      </tr>
      <tr>
        <td class="sae-string">Instance count:</td>
        <td class="sae-number">{data.count} ({pctFormat(data.pct)})</td>
      </tr>
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
