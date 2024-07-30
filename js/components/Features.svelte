<script lang="ts">
  import { scaleSequential } from "d3-scale";
  import { interpolateBlues } from "d3-scale-chromatic";
  import { format } from "d3-format";
  import { hcl } from "d3-color";
  import {
    feature_data,
    feature_index,
    sae_data,
  } from "../synced-state.svelte";
  import Histogram from "./vis/Histogram.svelte";

  const percentFormat = format(".3%");

  let maxActivation = $derived(
    Math.max(...feature_data.value.sequences.flatMap((seq) => seq.activation))
  );

  let color = $derived(
    scaleSequential().domain([0, maxActivation]).interpolator(interpolateBlues)
  );
</script>

<div class="wv-data-container">
  <div class="col">
    <select bind:value={feature_index.value}>
      {#each sae_data.value.alive_feature_indices as i}
        <option value={i}>
          {i}
        </option>
      {/each}
    </select>

    <div>
      This feature fires on {percentFormat(feature_data.value.firing_rate)} of tokens
    </div>

    <Histogram
      data={feature_data.value.activations_histogram}
      marginTop={20}
      marginRight={20}
      marginLeft={50}
      marginBottom={40}
      width={300}
      height={300}
      xAxisLabel={"Activation value"}
      yAxisLabel={"Token count"}
    />
  </div>

  <div class="col">
    <div class="sequences">
      {#each feature_data.value.sequences as seq}
        <div class="sequence">
          {#each seq.token as tok, i}
            <div
              class="token"
              style:background={color(seq.activation[i])}
              style:color={hcl(color(seq.activation[i])).l > 50
                ? "black"
                : "white"}
            >
              {tok}
            </div>
          {/each}
        </div>
      {/each}
    </div>
  </div>
</div>

<style>
  .wv-data-container {
    height: 100%;
    display: flex;
    flex-direction: row;
  }

  .col {
    flex: 1;
  }

  .sequences {
    display: flex;
    flex-direction: column;
  }

  .sequence + .sequence {
    padding: 0.25em;
  }

  .sequence {
    display: flex;
    white-space: pre;
  }
</style>
