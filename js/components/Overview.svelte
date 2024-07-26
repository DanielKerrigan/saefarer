<script lang="ts">
  import { sae_data } from "../synced-state.svelte";
  import Histogram from "./vis/Histogram.svelte";
</script>

<div class="wv-data-container">
  <Histogram
    data={sae_data.value.firing_rate_histogram}
    marginLeft={50}
    marginBottom={50}
    width={300}
    height={300}
  />
</div>

<style>
  .wv-data-container {
    height: 100%;
    display: flex;
    flex-direction: column;
  }
</style>
