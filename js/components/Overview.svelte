<script lang="ts">
  import { model_info, sae_data } from "../synced-state.svelte";
  import ConfusionMatrix from "./vis/ConfusionMatrix.svelte";
  import Histogram from "./vis/Histogram.svelte";
  import { format } from "d3-format";

  const percentFormat = format(".1%");

  const percentDead = $derived(
    sae_data.value.num_dead_features / sae_data.value.num_total_features,
  );

  const percentNonActivating = $derived(
    sae_data.value.num_non_activating_features /
      sae_data.value.num_total_features,
  );

  let maxHistHeight = $state(0);
  let maxHistWidth = $state(0);
  const histWidth = $derived(Math.min(maxHistWidth, maxHistHeight));
  const histHeight = $derived(Math.min(maxHistWidth, maxHistHeight));

  let maxCMHeight = $state(0);
  let maxCMWidth = $state(0);
  const cmWidth = $derived(Math.min(maxCMWidth, maxCMHeight));
  const cmHeight = $derived(Math.min(maxCMWidth, maxCMHeight));
</script>

<div class="sae-overview-container">
  <div class="sae-col">
    <div class="sae-header">Feature Activations</div>

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
        width={histWidth}
        height={histHeight}
        xAxisLabel={"lg activation rate →"}
        yAxisLabel={"↑ Feature count"}
      />
    </div>
  </div>

  <div class="sae-col">
    <div class="sae-header">Confusion Matrix</div>
    <div
      class="sae-vis"
      bind:offsetWidth={maxCMHeight}
      bind:offsetHeight={maxCMWidth}
    >
      <ConfusionMatrix
        cm={model_info.value.cm}
        width={cmWidth}
        height={cmHeight}
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

  .sae-header {
    font-weight: bold;
  }
</style>
