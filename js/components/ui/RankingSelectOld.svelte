<script lang="ts">
  import { model_info } from "../../synced-state.svelte";
  import ChainedSelect, {
    type Select,
    type Option,
  } from "./ChainedSelect.svelte";

  function getLabelOptions(): Option[] {
    const labels: Option[] = model_info.value.labels.map((d, i) => ({
      label: d,
      value: `${i}`,
      children: [],
    }));

    const options = [
      { label: "Any", value: "any", children: [] },
      { label: "Other", value: "other", children: [] },
      ...labels,
    ];

    return options;
  }

  const select: Select = {
    title: "Ranking",
    key: "ranking",
    defaultOptionIndex: 0,
    options: [
      {
        label: "Index",
        value: "index",
        children: [],
      },
      {
        label: "Max Act. Rate",
        value: "max_act_rate",
        children: [],
      },
      {
        label: "Min Act. Rate",
        value: "min_act_rate",
        children: [],
      },
      {
        label: "Label",
        value: "label",
        children: [
          {
            title: "True",
            key: "ground_truth",
            options: getLabelOptions(),
            defaultOptionIndex: 2,
          },
          {
            title: "Predicted",
            key: "predicted",
            options: getLabelOptions(),
            defaultOptionIndex: 0,
          },
        ],
      },
    ],
  };

  let choice = $state({});

  $inspect(choice);
</script>

<div>
  <ChainedSelect {select} bind:choice />
</div>

<style>
  div {
    display: flex;
    align-items: center;
    gap: 0.5em;
  }
</style>
