<script lang="ts">
  import Tabs from "./Tabs.svelte";
  import type { Tab } from "../types";
  import { height } from "../synced-state.svelte";
  import Overview from "./Overview.svelte";
  import Features from "./Features.svelte";

  let selectedTab: Tab = $state("overview");

  function changeTab(tab: Tab) {
    selectedTab = tab;
  }
</script>

<div class="sae-widget-container" style:height="{height.value}px">
  <Tabs {selectedTab} {changeTab} />

  <div class="sae-tab-content" class:sae-hide={selectedTab !== "overview"}>
    <Overview />
  </div>

  <div class="sae-tab-content" class:sae-hide={selectedTab !== "features"}>
    <Features />
  </div>
</div>

<style>
  .sae-widget-container {
    box-sizing: border-box;
    width: 100%;
    display: flex;
    flex-direction: column;
    font-size: 16px;
    border: 1px solid var(--gray-1);
    background-color: white;
    color: black;
  }

  .sae-tab-content {
    flex: 1;
    min-height: 0;
  }
</style>
