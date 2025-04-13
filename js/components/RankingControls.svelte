<script lang="ts">
  import { model_info, table_ranking_option } from "../synced-state.svelte";
  import type { RankingOption } from "../types";

  const rankingOptions: { label: string; value: RankingOption["kind"] }[] = [
    { label: "ID", value: "feature_id" },
    { label: "Act. Rate", value: "sequence_act_rate" },
    { label: "Confusion Matrix", value: "label" },
  ];

  const extraLabelOptions = [
    { label: "Any", value: "any" },
    { label: "Other", value: "other" },
  ];

  const labelOptions = $derived(
    model_info.value.labels.map((d, i) => ({
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
        true_label: "any",
        pred_label: "other",
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
</script>

<div class="sae-container">
  <label>
    <span class="sae-label">Ranking:</span>
    <select value={table_ranking_option.value.kind} onchange={onChangeRanking}>
      {#each rankingOptions as opt}
        <option value={opt.value}>{opt.label}</option>
      {/each}
    </select>
  </label>

  {#if table_ranking_option.value.kind === "label"}
    <label>
      <span class="sae-label">ŷ:</span>
      <select
        value={table_ranking_option.value.pred_label}
        onchange={(e) => onChangeLabel(e, "pred_label")}
      >
        <optgroup label="Extras">
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
      <span class="sae-label">y:</span>
      <select
        value={table_ranking_option.value.true_label}
        onchange={(e) => onChangeLabel(e, "true_label")}
      >
        <optgroup label="Extras">
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

  <div class="sae-feature-table-order">
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
</div>

<style>
  .sae-container {
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

  .sae-label {
    font-weight: 500;
  }

  .sae-feature-table-order {
    display: flex;
    gap: 0.5em;
  }
</style>
