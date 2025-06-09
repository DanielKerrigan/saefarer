<script lang="ts">
  import {
    dataset_info,
    table_min_act_rate,
    table_ranking_option,
  } from "../synced-state.svelte";
  import type { LabelRankingOption, RankingOption } from "../types";
  import InfoIcon from "./icons/InfoIcon.svelte";
  import TooltipButton from "./TooltipButton.svelte";

  const rankingOptions: { label: string; value: RankingOption["kind"] }[] = [
    { label: "ID", value: "feature_id" },
    { label: "Activation Rate", value: "sequence_act_rate" },
    { label: "Confusion Matrix", value: "label" },
  ];

  const extraLabelOptions = [
    { label: "Any", value: "any" },
    { label: "Different", value: "different" },
  ];

  const labelOptions = $derived(
    dataset_info.value.labels.map((d, i) => ({
      label: d,
      value: `${i}`,
    })),
  );

  function onChangeRanking(
    event: Event & { currentTarget: EventTarget & HTMLSelectElement },
  ) {
    const value = event.currentTarget.value;

    if (value === "feature_id") {
      table_ranking_option.value = {
        kind: "feature_id",
        descending: table_ranking_option.value.descending,
      };
    } else if (value === "sequence_act_rate") {
      table_ranking_option.value = {
        kind: "sequence_act_rate",
        descending: table_ranking_option.value.descending,
      };
    } else {
      table_ranking_option.value = {
        kind: "label",
        true_label: "different",
        pred_label: "any",
        descending: table_ranking_option.value.descending,
      };
    }
  }

  function onChangeLabel(
    event: Event & { currentTarget: EventTarget & HTMLSelectElement },
    key: "true_label" | "pred_label",
  ) {
    if (table_ranking_option.value.kind !== "label") {
      return;
    }

    const value = event.currentTarget.value;

    table_ranking_option.value = {
      ...table_ranking_option.value,
      [key]: value,
    };
  }

  function onChangeDirection(
    event: Event & { currentTarget: EventTarget & HTMLInputElement },
  ) {
    const value = event.currentTarget.value;

    table_ranking_option.value = {
      ...table_ranking_option.value,
      descending: value === "descending",
    };
  }

  let minActRateInputValue = $derived(table_min_act_rate.value * 100);

  function onMinActRateKeydown(
    event: KeyboardEvent & { currentTarget: EventTarget & HTMLInputElement },
  ) {
    if (event.key === "Enter") {
      updateMinActRate();
    }
  }

  function updateMinActRate() {
    table_min_act_rate.value = minActRateInputValue / 100;
  }

  function getCMRankingExlanation(rankingOption: LabelRankingOption): string {
    const { true_label, pred_label } = rankingOption;

    const prefix = "Your current selection ranks the features by";
    const poi = "the percentage of instances where";

    const any = "any";
    const diff = "different";

    const getName = (i: string) => dataset_info.value.labels[Number(i)];

    if (
      (true_label === any && pred_label === any) ||
      (true_label === diff && pred_label === diff)
    ) {
      return `${prefix} their ID. Try another combination!`;
    } else if (
      (true_label === any && pred_label === diff) ||
      (true_label === diff && pred_label === any)
    ) {
      return `${prefix} ${poi} the model is wrong.`;
    } else if (true_label === any) {
      return `${prefix} ${poi} the model predicts ${getName(pred_label)}.`;
    } else if (pred_label === any) {
      return `${prefix} ${poi} the true label is ${getName(true_label)}.`;
    } else if (true_label === diff) {
      return `${prefix} ${poi} the model incorrectly predicts ${getName(pred_label)}.`;
    } else if (pred_label === diff) {
      return `${prefix} ${poi} the true label is ${getName(true_label)} and the model is incorrect.`;
    } else if (pred_label === true_label) {
      return `${prefix} ${poi} the model correctly predicts ${getName(true_label)}.`;
    } else {
      return `${prefix} ${poi} the model predicts ${getName(pred_label)}, but the true label is ${getName(true_label)}.`;
    }
  }
</script>

<div class="sae-container">
  <div class="sae-control-row">
    <label>
      <span class="sae-title">Ranking:</span>
      <select
        value={table_ranking_option.value.kind}
        onchange={onChangeRanking}
      >
        {#each rankingOptions as opt}
          <option value={opt.value}>{opt.label}</option>
        {/each}
      </select>
    </label>
    {#if table_ranking_option.value.kind === "label"}
      <label>
        <span class="sae-title">Predicted label:</span>
        <select
          value={table_ranking_option.value.pred_label}
          onchange={(e) => onChangeLabel(e, "pred_label")}
        >
          <optgroup label="Wildcards">
            {#each extraLabelOptions as opt}
              <option value={opt.value}>{opt.label}</option>
            {/each}
          </optgroup>
          <optgroup label="Labels">
            {#each labelOptions as opt}
              <option value={opt.value}>{opt.label}</option>
            {/each}
          </optgroup>
        </select>
      </label>
      <label>
        <span class="sae-title">True label:</span>
        <select
          value={table_ranking_option.value.true_label}
          onchange={(e) => onChangeLabel(e, "true_label")}
        >
          <optgroup label="Wildcards">
            {#each extraLabelOptions as opt}
              <option value={opt.value}>{opt.label}</option>
            {/each}
          </optgroup>
          <optgroup label="Classes">
            {#each labelOptions as opt}
              <option value={opt.value}>{opt.label}</option>
            {/each}
          </optgroup>
        </select>
      </label>
    {/if}

    <TooltipButton position="bottom">
      {#snippet trigger()}
        <InfoIcon />
      {/snippet}
      {#snippet content()}
        <div class="sae-info">
          {#if table_ranking_option.value.kind === "feature_id"}
            The ID ranking orders the features by their index in the SAE. This
            essentially provides a random order.
          {:else if table_ranking_option.value.kind === "sequence_act_rate"}
            The activation rate ranking orders the features by the percentage of
            instances in the dataset that cause them to activate.
          {:else}
            <p>
              The confusion matrix ranking orders the features based on the
              model's predictions on the instances that cause the features to
              activate.
            </p>
            <p>
              {getCMRankingExlanation(table_ranking_option.value)}
            </p>
          {/if}
        </div>
      {/snippet}
    </TooltipButton>
  </div>

  <div class="sae-control-row">
    <div class="sae-feature-table-order">
      <span class="sae-title">Order:</span>
      <label>
        <input
          type="radio"
          name="direction"
          value={"ascending"}
          checked={!table_ranking_option.value.descending}
          onchange={onChangeDirection}
        />
        <span>Ascending</span>
      </label>
      <label>
        <input
          type="radio"
          name="direction"
          value={"descending"}
          checked={table_ranking_option.value.descending}
          onchange={onChangeDirection}
        />
        <span>Descending</span>
      </label>
    </div>
    <div class="sae-feature-table-min-act-rate">
      <label>
        <span class="sae-title">Min. activation rate:</span>
        <input
          type="number"
          bind:value={minActRateInputValue}
          onkeydown={onMinActRateKeydown}
          onblur={() => updateMinActRate()}
          step="0.0001"
          style:width="7em"
        />
        <span>%</span>
      </label>
    </div>
  </div>
</div>

<style>
  .sae-container {
    display: flex;
    flex-direction: column;
    gap: 0.5em;
  }

  .sae-control-row {
    display: flex;
    align-items: center;
    gap: 1em;
  }

  .sae-info {
    font-size: var(--text-sm);
    max-width: 32em;
  }

  .sae-title {
    font-weight: var(--font-medium);
  }

  label {
    display: flex;
    align-items: center;
    gap: 0.25em;
  }

  select {
    border: 1px solid var(--color-black);
    border-radius: 0.25em;
  }

  input {
    border: 1px solid var(--color-black);
    border-radius: 0.25em;
    padding: 0 0.25em;
  }

  .sae-feature-table-order {
    display: flex;
    align-items: center;
    gap: 0.5em;
  }
</style>
