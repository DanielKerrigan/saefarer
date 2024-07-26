<script lang="ts">
  import type { Tab } from "../types";

  let {
    selectedTab,
    changeTab,
  }: { selectedTab: Tab; changeTab: (tab: Tab) => void } = $props();

  const tabs: { value: Tab; title: string }[] = [
    { value: "overview", title: "Overview" },
    { value: "features", title: "Features" },
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
    padding: 0.25em;
    background-color: var(--gray-1);
  }

  button {
    border: none;
    border-radius: 0;
    background-color: transparent;
    color: black;
  }

  button:hover:enabled {
    color: black;
    background-color: var(--gray-2);
  }

  button:active:enabled {
    color: black;
    background-color: var(--gray-3);
  }

  button.tab-selected {
    border-bottom: 1px solid black;
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
