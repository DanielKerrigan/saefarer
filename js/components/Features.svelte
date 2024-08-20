<script lang="ts">
  import { scaleSequential } from "d3-scale";
  import { interpolateBlues } from "d3-scale-chromatic";
  import { format } from "d3-format";
  import { hcl } from "d3-color";
  import { feature_data, feature_id, sae_data } from "../synced-state.svelte";
  import Histogram from "./vis/Histogram.svelte";
  import LineChart from "./vis/LineChart.svelte";
  import Tooltip from "./Tooltip.svelte";

  const percentFormat = format(".3%");

  let color = $derived(
    scaleSequential()
      .domain([0, feature_data.value.max_activation])
      .interpolator(interpolateBlues)
  );

  let chosenInterval = $state(Object.keys(feature_data.value.sequences)[0]);

  let wrapSequences = $state(false);

  let tooltipToShow = $state({ sequence: -1, token: -1 });
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
        bandY0={sae_data.value.cumsum_percent_l1_norm_range.mins}
        bandY1={sae_data.value.cumsum_percent_l1_norm_range.maxs}
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
    <div class="sae-header">Example Activations</div>

    <div class="sae-sequences-controls">
      <select bind:value={chosenInterval}>
        {#each Object.keys(feature_data.value.sequences) as intervalName}
          <option value={intervalName}>
            {intervalName}
          </option>
        {/each}
      </select>

      <label>
        <input type="checkbox" bind:checked={wrapSequences} />
        <span>Wrap</span>
      </label>
    </div>

    <div class="sae-sequences">
      {#each feature_data.value.sequences[chosenInterval] as seq, seqIndex}
        <div
          class="sae-sequence"
          style:flex-wrap={wrapSequences ? "wrap" : "nowrap"}
        >
          {#each seq.token as token, tokIndex}
            {@const act = seq.activation[tokIndex]}
            <!-- TODO: do this properly -->
            <!-- svelte-ignore a11y_no_static_element_interactions -->
            <div
              class="sae-token"
              style:background={act > 0 ? color(act) : "white"}
              style:color={hcl(color(act)).l > 50 ? "black" : "white"}
              style:font-weight={tokIndex === seq.max_index ? "bold" : "normal"}
              onmouseenter={() =>
                (tooltipToShow = { sequence: seqIndex, token: tokIndex })}
              onmouseleave={() => (tooltipToShow = { sequence: -1, token: -1 })}
            >
              <span class="sae-token-name">{token}</span>
              {#if tooltipToShow.token === tokIndex && tooltipToShow.sequence === seqIndex}
                <Tooltip sequence={seq} index={tokIndex} />
              {/if}
            </div>
          {/each}
        </div>
      {/each}
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

  .sae-sequences {
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    gap: 0.25em;
  }

  .sae-sequence + .sae-sequence {
    padding-top: 0.25em;
    border-top: 2px solid var(--gray-1);
  }

  .sae-sequence {
    display: flex;
  }

  .sae-token {
    position: relative;
  }

  .sae-token-name {
    white-space: pre;
  }

  .sae-header {
    font-weight: bold;
  }

  select {
    align-self: flex-start;
  }

  label {
    display: flex;
    align-items: center;
    gap: 0.25em;
  }

  .sae-sequences-controls {
    display: flex;
    justify-content: space-between;
  }
</style>
