<script lang="ts">
  import { sae_data } from "../synced-state.svelte";
  import FeatureProjectionScatter from "./vis/FeatureProjectionScatter.svelte";
  import Histogram from "./vis/Histogram.svelte";
  import { format } from "d3-format";

  const percentFormat = format(".1%");
  const percentDead = $derived(
    sae_data.value.num_dead_features /
      (sae_data.value.num_dead_features + sae_data.value.num_alive_features)
  );

  let borderBoxSizeLeft: ResizeObserverSize[] | undefined = $state();
  const leftWidth = $derived(
    borderBoxSizeLeft ? borderBoxSizeLeft[0].inlineSize : 0
  );

  let borderBoxSizeRight: ResizeObserverSize[] | undefined = $state();
  const rightWidth = $derived(
    borderBoxSizeRight ? borderBoxSizeRight[0].inlineSize : 0
  );
  const height = $derived(
    borderBoxSizeRight ? borderBoxSizeRight[0].blockSize : 0
  );
</script>

<div class="sae-overview-container">
  <div class="sae-left" bind:borderBoxSize={borderBoxSizeLeft}>
    <div class="sae-section">
      <div class="sae-header">Feature Activation Rates</div>
      <div>{percentFormat(percentDead)} of features are dead.</div>
      <Histogram
        data={sae_data.value.activation_rate_histogram}
        marginTop={20}
        marginRight={20}
        marginLeft={50}
        marginBottom={40}
        width={leftWidth}
        height={200}
        xAxisLabel={"log_10 activation rate"}
        yAxisLabel={"Feature count"}
      />
    </div>

    <div class="sae-section">
      <div class="sae-header">Feature Dimensionality</div>
      <Histogram
        data={sae_data.value.dimensionality_histogram}
        marginTop={20}
        marginRight={20}
        marginLeft={50}
        marginBottom={40}
        width={leftWidth}
        height={200}
        xAxisLabel={"Number of dimensions to explain majority of L1 norm"}
        yAxisLabel={"Feature count"}
      />
    </div>
  </div>

  <div class="sae-right" bind:borderBoxSize={borderBoxSizeRight}>
    <div class="sae-section">
      <div class="sae-header">Feature Projection</div>
      <FeatureProjectionScatter
        data={sae_data.value.feature_projection}
        marginTop={2}
        marginRight={2}
        marginLeft={2}
        marginBottom={2}
        width={rightWidth}
        {height}
      />
    </div>
  </div>
</div>

<style>
  .sae-overview-container {
    height: 100%;
    display: flex;
    padding: 1em;
    gap: 1em;
  }

  .sae-left {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 0.5em;
  }

  .sae-right {
    flex: 2;
    display: flex;
    flex-direction: column;
  }

  .sae-section {
    display: flex;
    flex-direction: column;
    gap: 0.25em;
  }

  .sae-header {
    font-weight: bold;
  }
</style>
