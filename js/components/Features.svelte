<script lang="ts">
  import { scaleSequential } from "d3-scale";
  import { interpolateBlues } from "d3-scale-chromatic";
  import { format } from "d3-format";
  import { hcl } from "d3-color";
  import { feature_data, feature_id, sae_data } from "../synced-state.svelte";
  import Histogram from "./vis/Histogram.svelte";
  import LineChart from "./vis/LineChart.svelte";

  const percentFormat = format(".3%");

  let color = $derived(
    scaleSequential()
      .domain([0, feature_data.value.max_activation])
      .interpolator(interpolateBlues)
  );

  let chosenInterval = $state(Object.keys(feature_data.value.sequences)[0]);
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
      <div class="sae-header">Dimensionality</div>

      <LineChart
        xs={feature_data.value.cumsum_percent_l1_norm.n_neurons}
        ys={feature_data.value.cumsum_percent_l1_norm.cum_sum}
        marginTop={20}
        marginRight={20}
        marginLeft={50}
        marginBottom={40}
        width={300}
        height={200}
        xAxisLabel={"Number of dimensions"}
        yAxisLabel={"Percent of L1 norm"}
      />
    </div>
  </div>

  <div class="sae-right">
    <div class="sae-section">
      <div class="sae-header">Example Activations</div>

      <select bind:value={chosenInterval}>
        {#each Object.keys(feature_data.value.sequences) as intervalName}
          <option value={intervalName}>
            {intervalName}
          </option>
        {/each}
      </select>

      <div class="sae-sequences">
        {#each feature_data.value.sequences[chosenInterval] as seq}
          <div class="sae-sequence">
            {#each seq.token as tok, i}
              <div
                class="sae-token"
                style:background={color(seq.activation[i])}
                style:color={hcl(color(seq.activation[i])).l > 50
                  ? "black"
                  : "white"}
                style:font-weight={i === seq.max_index ? "bold" : "normal"}
              >
                {tok}
              </div>
            {/each}
          </div>
        {/each}
      </div>
    </div>
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
    flex: 1;
    display: flex;
    flex-direction: column;
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

  .sae-sequences {
    display: flex;
    flex-direction: column;
  }

  .sae-sequence + .sae-sequence {
    padding-top: 0.25em;
  }

  .sae-sequence {
    overflow: auto;
    display: flex;
    white-space: pre;
  }

  .sae-header {
    font-weight: bold;
  }

  select {
    align-self: flex-start;
  }
</style>
