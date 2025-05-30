<script lang="ts">
  import { model_info, sae_data } from "../synced-state.svelte";
  import ConfusionMatrix from "./vis/ConfusionMatrix.svelte";
  import Histogram from "./vis/Histogram.svelte";
  import {
    activationRateLogFormat,
    activationRatePctFormat,
    countFormat,
    getSizeWithAspectRatio,
    getSizeWithAspectRatioMargins,
    percentFormat,
  } from "./vis/vis-utils";

  const percentDead = $derived(
    sae_data.value.n_dead_features / sae_data.value.n_total_features,
  );

  const percentNonActivating = $derived(
    sae_data.value.n_non_activating_features / sae_data.value.n_total_features,
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
    <div>
      {percentFormat(percentDead)} of features died during training.
    </div>

    <div>
      {percentFormat(percentNonActivating)} of features did not activate during analysis.
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

  <div class="sae-col">
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
    gap: 0.5em;
    height: 100%;
  }

  .sae-vis {
    flex: 1;
    min-height: 0;
  }
</style>
