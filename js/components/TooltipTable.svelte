<script lang="ts">
  let {
    data,
  }: {
    data: { key: string; value: string }[];
  } = $props();
</script>

<table>
  <tbody>
    {#each data as { key, value }}
      <tr>
        <td>{key}</td>
        <td>{value}</td>
      </tr>
    {/each}
  </tbody>
</table>

<style>
  table {
    border-collapse: collapse;
    font-size: var(--text-sm);
  }

  td {
    padding: 0em 0.5em 0.25em 0em;
    line-height: 1;
    vertical-align: middle;
    font-variant-numeric: lining-nums tabular-nums;
  }

  /* no right padding for last column in table */
  tr > td:last-of-type {
    padding-right: 0;
  }

  /* no bottom padding for last row in table */
  tbody > tr:last-of-type > td {
    padding-bottom: 0;
  }

  table tr td:first-child {
    color: var(--color-neutral-700);
    font-weight: var(--font-normal);
    text-align: right;
  }

  table tr td:nth-child(2) {
    color: var(--color-black);
    font-weight: var(--font-medium);
    text-align: left;
  }
</style>
