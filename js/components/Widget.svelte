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

<div class="wv-widget-container" style:height="{height.value}px">
  <Tabs {selectedTab} {changeTab} />

  <div class="wv-tab-content" class:wv-hide={selectedTab !== "overview"}>
    <Overview />
  </div>

  <div class="wv-tab-content" class:wv-hide={selectedTab !== "features"}>
    <Features />
  </div>
</div>

<style>
  .wv-widget-container {
    box-sizing: border-box;
    width: 100%;
    display: flex;
    flex-direction: column;
    font-size: 16px;
    border: 1px solid var(--gray-1);
    background-color: white;
    color: black;
  }

  .wv-tab-content {
    flex: 1;
    min-height: 0;
  }
</style>
