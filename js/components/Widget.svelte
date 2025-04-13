<script lang="ts">
  import Tabs from "./Tabs.svelte";
  import type { Tab } from "../types";
  import {
    model_info,
    base_font_size,
    detail_feature_id,
    height,
  } from "../synced-state.svelte";
  import Overview from "./Overview.svelte";
  import FeatureTable from "./FeatureTable.svelte";
  import FeatureDetail from "./FeatureDetail.svelte";
  import { rootDiv } from "../state.svelte";
  import { scaleOrdinal } from "d3-scale";
  import { schemeObservable10 } from "d3-scale-chromatic";

  let selectedTab: Tab = $state("overview");

  function changeTab(tab: Tab) {
    selectedTab = tab;
  }

  function onClickFeature(feature_id: number) {
    detail_feature_id.value = feature_id;
    selectedTab = "detail";
  }

  const labelColor = $derived(
    scaleOrdinal<number, string>()
      .domain(model_info.value.label_indices)
      .range(schemeObservable10),
  );
</script>

<div
  class="sae-widget-container"
  style:height="{height.value}px"
  style:font-size="{base_font_size.value}px"
  bind:this={rootDiv.value}
>
  <div class="sae-tabs-container">
    <Tabs {selectedTab} {changeTab} />
  </div>

  <div class="sae-tab-content">
    {#if selectedTab === "overview"}
      <Overview />
    {:else if selectedTab === "table"}
      <FeatureTable {labelColor} {onClickFeature} />
    {:else}
      <FeatureDetail {labelColor} />
    {/if}
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

  .sae-tabs-container {
    flex: none;
  }

  .sae-tab-content {
    flex: 1;
    min-height: 0;
    padding: 0.5em;
  }
</style>
