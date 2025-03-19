<script lang="ts">
  import { scaleSequential } from "d3-scale";
  import { interpolateBlues } from "d3-scale-chromatic";
  import { format } from "d3-format";
  import { feature_data, feature_id, sae_data } from "../synced-state.svelte";
  import Histogram from "./vis/Histogram.svelte";
  import FeatureTokenSequences from "./FeatureTokenSequences.svelte";
  import MarginalEffectsPlot from "./vis/MarginalEffectsPlot.svelte";

  const percentFormat = format(".3%");

  let color = $derived(
    scaleSequential()
      .domain([0, feature_data.value.max_act])
      .interpolator(interpolateBlues),
  );
</script>

<div class="sae-features-container">
  <div class="sae-left">
    <div class="sae-header">Feature Selection</div>

    <select bind:value={feature_id.value}>
      {#each sae_data.value.alive_feature_ids as i}
        <option value={i}>
          {i}
        </option>
      {/each}
    </select>
  </div>

  <div class="sae-middle">
    <div class="sae-section">
      <div class="sae-header">Activations</div>

      <div>
        Activation rate: {percentFormat(feature_data.value.activation_rate)} of tokens
      </div>

      <Histogram
        data={feature_data.value.activations_histogram}
        marginTop={20}
        marginRight={20}
        marginLeft={50}
        marginBottom={40}
        width={300}
        height={200}
        xAxisLabel={"Activation value"}
        yAxisLabel={"Token count"}
      />
    </div>

    <div class="sae-section">
      <div class="sae-header">Marginal Effects</div>

      <MarginalEffectsPlot
        data={feature_data.value.marginal_effects}
        marginTop={20}
        marginRight={20}
        marginLeft={50}
        marginBottom={40}
        width={300}
        height={200}
        xAxisLabel={"Activation value"}
        yAxisLabel={"Average probability"}
      />
    </div>
  </div>

  <div class="sae-right">
    <div class="sae-header">Example Activations</div>

    <FeatureTokenSequences {color} />
  </div>
</div>

<style>
  .sae-features-container {
    height: 100%;
    display: flex;
    flex-direction: row;
    padding: 1em;
    gap: 1em;
  }

  .sae-left {
    flex: 0 0 200px;
    display: flex;
    flex-direction: column;
    gap: 0.5em;
  }

  .sae-middle {
    min-width: 300px;
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
  }

  .sae-right {
    min-width: 0;
    flex: 2;
    display: flex;
    flex-direction: column;
    gap: 0.25em;
  }

  .sae-section {
    display: flex;
    flex-direction: column;
    gap: 0.25em;
  }

  .sae-header {
    font-weight: bold;
  }

  select {
    align-self: flex-start;
  }
</style>
