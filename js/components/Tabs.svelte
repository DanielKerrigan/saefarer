<script lang="ts">
  import type { Tab } from "../types";

  let {
    selectedTab,
    changeTab,
  }: { selectedTab: Tab; changeTab: (tab: Tab) => void } = $props();

  const tabs: { value: Tab; title: string }[] = [
    { value: "overview", title: "Overview" },
    { value: "table", title: "Table" },
    { value: "detail", title: "Detail" },
  ];
</script>

<div>
  <ul>
    {#each tabs as { value, title }}
      <li>
        <button
          class:tab-selected={value === selectedTab}
          onclick={() => changeTab(value)}
        >
          {title}
        </button>
      </li>
    {/each}
  </ul>
</div>

<style>
  div {
    display: flex;
    padding: 0.25em 0.5em;
    background-color: var(--color-black);
  }

  button {
    border: none;
    border-radius: 0;
    background-color: transparent;
    color: var(--color-white);
  }

  button:hover:enabled {
    color: var(--color-white);
    background-color: var(--color-neutral-700);
  }

  button:active:enabled {
    color: var(--color-white);
    background-color: var(--color-neutral-500);
  }

  button.tab-selected {
    border-bottom: 1px solid var(--color-white);
  }

  ul {
    list-style: none;
  }

  li {
    display: inline-flex;
  }

  li + li {
    margin-left: 0.5em;
  }
</style>
