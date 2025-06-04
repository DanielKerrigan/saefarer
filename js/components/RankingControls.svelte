<script lang="ts">
  import {
    dataset_info,
    table_min_act_rate,
    table_ranking_option,
  } from "../synced-state.svelte";
  import type { RankingOption } from "../types";

  const rankingOptions: { label: string; value: RankingOption["kind"] }[] = [
    { label: "ID", value: "feature_id" },
    { label: "Act. Rate", value: "sequence_act_rate" },
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
</script>

<div class="sae-container">
  <div class="sae-control-row">
    <label>
      <span style:font-weight="var(--font-medium)">Ranking:</span>
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
        <span style:font-weight="var(--font-medium)">Predicted label:</span>
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
        <span style:font-weight="var(--font-medium)">True label:</span>
        <select
          value={table_ranking_option.value.true_label}
          onchange={(e) => onChangeLabel(e, "true_label")}
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
    {/if}
  </div>

  <div class="sae-control-row">
    <div class="sae-feature-table-order">
      <span style:font-weight="var(--font-medium)">Order:</span>
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
        <span style:font-weight="var(--font-medium)">Min. activation rate:</span
        >
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
