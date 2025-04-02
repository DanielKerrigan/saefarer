<script lang="ts">
  import Tabs from "./Tabs.svelte";
  import type { Tab } from "../types";
  import { base_font_size, height } from "../synced-state.svelte";
  import Overview from "./Overview.svelte";
  import Table from "./Table.svelte";
  import Detail from "./Detail.svelte";
  import { rootDiv } from "../state.svelte";

  let selectedTab: Tab = $state("overview");

  function changeTab(tab: Tab) {
    selectedTab = tab;
  }
</script>

<div
  class="sae-widget-container"
  style:height="{height.value}px"
  style:font-size="{base_font_size.value}px"
  bind:this={rootDiv.value}
>
  <div class="tabs-container">
    <Tabs {selectedTab} {changeTab} />
  </div>

  <div class="sae-tab-content" class:sae-hide={selectedTab !== "overview"}>
    <Overview />
  </div>

  <div class="sae-tab-content" class:sae-hide={selectedTab !== "table"}>
    <Table />
  </div>

  <div class="sae-tab-content" class:sae-hide={selectedTab !== "detail"}>
    <Detail />
  </div>
</div>

<style>
  .sae-widget-container {
    box-sizing: border-box;
    position: relative;
    width: 100%;
    display: flex;
    flex-direction: column;
    border: 1px solid var(--color-black);
    background-color: var(--color-white);
    color: var(--color-black);
  }

  .tabs-container {
    flex: none;
  }

  .sae-tab-content {
    flex: 1;
    min-height: 0;
    padding: 0.5em;
  }
</style>
