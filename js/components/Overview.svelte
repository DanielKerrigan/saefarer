<script lang="ts">
  import { dataset_info, model_info, sae_data } from "../synced-state.svelte";
  import HelpIcon from "./icons/HelpIcon.svelte";
  import TooltipButton from "./TooltipButton.svelte";
  import ConfusionMatrix from "./vis/ConfusionMatrix.svelte";
  import Histogram from "./vis/Histogram.svelte";
  import {
    activationRateLogFormat,
    activationRatePctFormat,
    countFormat,
    getSizeWithAspectRatio,
    getSizeWithAspectRatioMargins,
    logLossFormat,
    percentFormat,
    siFormat,
  } from "./vis/vis-utils";

  const totalInactiveFeatures = $derived(
    sae_data.value.n_non_activating_features + sae_data.value.n_dead_features,
  );

  const percentInactiveFeatures = $derived(
    totalInactiveFeatures / sae_data.value.n_total_features,
  );

  let maxHistWidth = $state(0);
  let maxHistHeight = $state(0);
  const histSize = $derived(
    getSizeWithAspectRatio(maxHistWidth, maxHistHeight, 1.6),
  );

  const cmMarginTop = 8;
  const cmMarginRight = 88;
  const cmMarginBottom = 80;
  const cmMarginLeft = 80;

  let maxCMWidth = $state(0);
  let maxCMHeight = $state(0);
  const cmSize = $derived(
    getSizeWithAspectRatioMargins(
      maxCMWidth,
      maxCMHeight,
      1,
      cmMarginTop,
      cmMarginRight,
      cmMarginBottom,
      cmMarginLeft,
    ),
  );
</script>

<div class="sae-overview-container">
  <div class="sae-col">
    <div class="sae-section">
      <div class="sae-section-header">Summary</div>
      <div class="sae-table-container">
        <table>
          <thead>
            <tr>
              <th colspan="2">Dataset</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Instances</td>
              <td>{siFormat(dataset_info.value.n_sequences)}</td>
            </tr>
            <tr>
              <td>Tokens</td>
              <td>{siFormat(dataset_info.value.n_tokens)}</td>
            </tr>
          </tbody>
        </table>
        <table>
          <thead>
            <tr>
              <th colspan="2">Model</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Error rate</td>
              <td>{percentFormat(model_info.value.cm.error_pct)}</td>
            </tr>
            <tr>
              <td>Log loss</td>
              <td>{logLossFormat(model_info.value.log_loss)}</td>
            </tr>
          </tbody>
        </table>
        <table>
          <thead>
            <tr>
              <th colspan="2">SAE</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Total features</td>
              <td>{countFormat(sae_data.value.n_total_features)}</td>
            </tr>
            <tr>
              <td>Inactive features</td>
              <td
                >{countFormat(totalInactiveFeatures)} ({percentFormat(
                  percentInactiveFeatures,
                )})</td
              >
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="sae-section" style:flex="1">
      <div class="sae-section-header-row">
        <div class="sae-section-header">
          Feature activation rate distribution
        </div>

        <TooltipButton position="right">
          {#snippet trigger()}
            <HelpIcon />
          {/snippet}
          {#snippet content()}
            <div class="sae-info">
              This histogram shows how often the features in the SAE activate.
              The activation rate is the percentage of instances that cause a
              feature to activate. Note that the x-axis uses a log scale.
            </div>
          {/snippet}
        </TooltipButton>
      </div>
      <div
        class="sae-vis"
        bind:offsetWidth={maxHistWidth}
        bind:offsetHeight={maxHistHeight}
      >
        <Histogram
          data={sae_data.value.sequence_act_rate_histogram}
          marginTop={20}
          marginRight={20}
          marginLeft={50}
          marginBottom={40}
          width={histSize.width}
          height={histSize.height}
          xAxisLabel={"lg activation rate →"}
          yAxisLabel={"↑ Feature count"}
          tooltipData={[
            {
              key: "Feature count",
              value: (_x1, _x2, y) => countFormat(y),
            },
            {
              key: "Activation rate",
              value: (x1, x2, _y) =>
                `${activationRatePctFormat(10 ** x1)} to ${activationRatePctFormat(10 ** x2)}`,
            },
            {
              key: "Log 10 act. rate",
              value: (x1, x2, _y) =>
                `${activationRateLogFormat(x1)} to ${activationRateLogFormat(x2)}`,
            },
          ]}
        />
      </div>
    </div>
  </div>

  <div class="sae-col">
    <div class="sae-section" style:flex="1">
      <div class="sae-section-header">Confusion Matrix</div>
      <div
        class="sae-vis"
        bind:offsetWidth={maxCMWidth}
        bind:offsetHeight={maxCMHeight}
      >
        <ConfusionMatrix
          cm={model_info.value.cm}
          legend={"vertical"}
          width={cmSize.width}
          height={cmSize.height}
          marginTop={cmMarginTop}
          marginRight={cmMarginRight}
          marginBottom={cmMarginBottom}
          marginLeft={cmMarginLeft}
        />
      </div>
    </div>
  </div>
</div>

<style>
  .sae-overview-container {
    height: 100%;
    display: flex;
    gap: 1em;
  }

  .sae-col {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 1em;
    height: 100%;
  }

  .sae-vis {
    flex: 1;
    min-height: 0;
  }

  .sae-section {
    display: flex;
    flex-direction: column;
    gap: 0.5em;
    min-height: 0;
  }

  .sae-section-header-row {
    display: flex;
    gap: 0.25em;
    align-items: center;
  }

  .sae-section-header {
    font-weight: var(--font-medium);
  }

  .sae-table-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5em;
  }

  table {
    border-collapse: collapse;
    align-self: flex-start;
    text-align: left;
    border: 2px solid var(--color-neutral-300);
  }

  thead {
    border-bottom: 2px solid var(--color-black);
    background-color: var(--color-neutral-200);
  }

  th {
    font-weight: var(--font-medium);
  }

  td {
    border-bottom: 2px solid var(--color-neutral-300);
  }

  th,
  td {
    padding: 0.25em 0.5em 0.25em 0.25em;
    line-height: 1;
    vertical-align: middle;
    font-variant-numeric: lining-nums tabular-nums;
  }

  table tr td:first-child {
    text-align: left;
  }

  table tr td:nth-child(2) {
    text-align: right;
  }

  .sae-info {
    font-size: var(--text-sm);
    max-width: 24em;
  }
</style>
