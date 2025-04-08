<script lang="ts">
  import { scaleSequential } from "d3-scale";
  import { interpolateBuPu } from "d3-scale-chromatic";
  import { format } from "d3-format";
  import { detail_feature, detail_feature_id } from "../synced-state.svelte";
  import FeatureTokenSequences from "./FeatureTokenSequences.svelte";
  import MarginalEffectsPlot from "./vis/MarginalEffectsPlot.svelte";
  import ConfusionMatrix from "./vis/ConfusionMatrix.svelte";

  const percentFormat = format(".3%");

  let color = $derived(
    scaleSequential()
      .domain([0, detail_feature.value.max_act])
      .interpolator(interpolateBuPu),
  );

  let featureIdInputValue = $derived(detail_feature_id.value);

  function onClickGo() {
    detail_feature_id.value = featureIdInputValue;
  }
</script>

<div class="sae-container">
  <div class="sae-controls">
    <div class="sae-feature-input">
      <label>
        <span>Feature</span>
        <input type="number" bind:value={featureIdInputValue} />
      </label>
      <button onclick={onClickGo}>Go</button>
    </div>
  </div>
  <div class="sae-main">
    <div class="sae-left">
      <div class="sae-section">
        <div class="sae-header">Prediction vs. Activation</div>

        <div>
          Activation rate: {percentFormat(
            detail_feature.value.sequence_act_rate,
          )}
          of instances
        </div>

        <MarginalEffectsPlot
          marginalEffects={detail_feature.value.marginal_effects}
          distribution={detail_feature.value.sequence_acts_histogram}
          marginTop={20}
          marginRight={20}
          marginLeft={50}
          marginBottom={40}
          width={300}
          height={200}
          xAxisLabel={"Activation value →"}
          yAxisLabel={"Mean predicted probability →"}
        />
      </div>
    </div>

    <div class="sae-middle">
      <div class="sae-section">
        <div class="sae-header">Confusion Matrix</div>

        <ConfusionMatrix
          cm={detail_feature.value.cm}
          width={300}
          height={300}
        />
      </div>
    </div>

    <div class="sae-right">
      <div class="sae-header">Example Activations</div>

      <FeatureTokenSequences {color} />
    </div>
  </div>
</div>

<style>
  .sae-container {
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 1em;
  }

  .sae-controls {
    display: flex;
  }

  .sae-main {
    flex: 1;
    display: flex;
    flex-direction: row;
    gap: 1em;
  }

  .sae-left {
    min-width: 300px;
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
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
    flex: 1;
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

  .sae-feature-input {
    display: flex;
    align-items: center;
    gap: 0.25em;
  }

  .sae-feature-input label {
    display: flex;
    align-items: center;
    gap: 0.25em;
  }

  .sae-feature-input label span {
    font-weight: bold;
  }

  .sae-feature-input label input {
    align-self: flex-start;
    border: 1px solid var(--color-neutral-400);
    padding: 0.25em 0.5em;
    width: 6em;
  }

  .sae-feature-input button {
    padding: 0.25em 0.5em;
  }
</style>
