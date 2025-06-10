<script lang="ts">
  import {
    table_page_index,
    max_table_page_index,
  } from "../synced-state.svelte";
  import { clamp } from "../utils";

  // plus one since index starts at 0, but ui starts at 1
  let pageNumberInputValue = $derived(table_page_index.value + 1);

  const maxNumDigits = $derived(Math.log10(max_table_page_index.value + 1) + 1);

  function goToPage(i: number) {
    const index = clamp(i, 0, max_table_page_index.value);

    // If we're at the first page and the user types in a negative number,
    // we want to reset the input value to 0. table_page_index is already 0,
    // so setting it to 0 again will not update pageIndexInputValue

    if (index === table_page_index.value) {
      pageNumberInputValue = index + 1;
    } else {
      table_page_index.value = index;
    }
  }

  function inputOnKeyDown(
    event: KeyboardEvent & { currentTarget: EventTarget & HTMLInputElement },
  ) {
    if (event.key === "Enter") {
      goToPage(pageNumberInputValue - 1);
    }
  }
</script>

<div class="sae-page-container">
  <button
    disabled={table_page_index.value <= 0}
    onclick={() => goToPage(pageNumberInputValue - 2)}
  >
    ←
  </button>

  <div class="sae-page-select">
    <span>Page</span>
    <input
      type="number"
      bind:value={pageNumberInputValue}
      onkeydown={inputOnKeyDown}
      onblur={() => goToPage(pageNumberInputValue - 1)}
      style:width="{maxNumDigits}em"
    />
    <span>/</span>
    <span>{max_table_page_index.value + 1}</span>
  </div>

  <button
    disabled={table_page_index.value >= max_table_page_index.value}
    onclick={() => goToPage(pageNumberInputValue)}
  >
    →
  </button>
</div>

<style>
  .sae-page-container {
    display: flex;
    gap: 0.5em;
    align-items: center;
  }

  .sae-page-select {
    display: flex;
    gap: 0.25em;
    align-items: center;
  }

  input[type="number"]::-webkit-inner-spin-button,
  input[type="number"]::-webkit-outer-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }

  input {
    border: 1px solid var(--color-black);
    padding: 0 0.25em;
  }

  button {
    padding: 0.25em;
  }
</style>
