<script lang="ts" module>
  export type Option = {
    label: string;
    value: string;
    children: Select[];
  };

  export type Select = {
    title: string;
    key: string;
    defaultOptionIndex: number;
    options: Option[];
  };

  // export type Choice = {
  //   key: string;
  //   value: string;
  //   children: Choice[];
  // };

  export type Choice = Record<
    string,
    {
      value: string;
      children: Choice;
    }
  >;
</script>

<script lang="ts">
  import { onMount } from "svelte";

  import Self from "./ChainedSelect.svelte";

  let { select, choice = $bindable() }: { select: Select; choice: Choice } =
    $props();

  let option = $state(select.options[select.defaultOptionIndex]);

  function onchange() {
    choice[select.key] = {
      value: option.value,
      children: {},
    };
  }

  onMount(() => {
    onchange();
  });

  $inspect(option);
</script>

<label>
  <span>{select.title}:</span>
  <select bind:value={option} {onchange}>
    {#each select.options as opt}
      <option value={opt}>{opt.label}</option>
    {/each}
  </select>
</label>

{#each option.children as child}
  <Self select={child} bind:choice={choice[select.key].children} />
{/each}

<style>
  label {
    display: flex;
    align-items: center;
    gap: 0.25em;
  }

  select {
    border: 1px solid var(--color-black);
  }
</style>
