<script lang="ts">
  import Tabs from "./Tabs.svelte";
  import type { Tab } from "../types";
  import {
    font_sizes,
    detail_feature_id,
    height,
  } from "../synced-state.svelte";
  import Overview from "./Overview.svelte";
  import FeatureTable from "./FeatureTable.svelte";
  import FeatureDetail from "./FeatureDetail.svelte";

  let selectedTab: Tab = $state("overview");

  function changeTab(tab: Tab) {
    selectedTab = tab;
  }

  function onClickFeature(feature_id: number) {
    detail_feature_id.value = feature_id;
    selectedTab = "detail";
  }
</script>

<div
  class="sae-widget-container"
  style:height="{height.value}px"
  style:--text-xs="{font_sizes.xs}px"
  style:--text-sm="{font_sizes.sm}px"
  style:--text-base="{font_sizes.base}px"
  style:--text-lg="{font_sizes.lg}px"
  style:--text-xl="{font_sizes.xl}px"
>
  <div class="sae-tabs-container">
    <Tabs {selectedTab} {changeTab} />
  </div>

  <div class="sae-tab-content">
    {#if selectedTab === "overview"}
      <Overview />
    {:else if selectedTab === "table"}
      <FeatureTable {onClickFeature} />
    {:else}
      <FeatureDetail />
    {/if}
  </div>
</div>

<style>
  .sae-widget-container {
    contain: layout;
    position: relative;
    width: 100%;
    display: flex;
    flex-direction: column;
    border: 1px solid var(--color-black);
    background-color: var(--color-white);
    color: var(--color-black);
    font-size: var(--text-base);
  }

  .sae-tabs-container {
    flex: none;
  }

  .sae-tab-content {
    flex: 1;
    min-height: 0;
    padding: 0.5em;
  }
</style>
