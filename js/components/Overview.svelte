<script lang="ts">
  import { sae_data } from "../synced-state.svelte";
  import FeatureProjectionScatter from "./vis/FeatureProjectionScatter.svelte";
  import Histogram from "./vis/Histogram.svelte";
</script>

<div class="wv-data-container">
  <Histogram
    data={sae_data.value.firing_rate_histogram}
    marginTop={20}
    marginRight={20}
    marginLeft={50}
    marginBottom={40}
    width={300}
    height={300}
    xAxisLabel={"log_10 activation rate"}
    yAxisLabel={"Feature count"}
  />

  <FeatureProjectionScatter
    data={sae_data.value.feature_projection}
    marginTop={20}
    marginRight={20}
    marginLeft={50}
    marginBottom={40}
    width={300}
    height={300}
    xAxisLabel={"x"}
    yAxisLabel={"y"}
  />
</div>

<style>
  .wv-data-container {
    height: 100%;
    display: flex;
    padding: 1em;
  }
</style>
