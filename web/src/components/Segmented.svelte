<script lang="ts" generics="T extends string">
  interface Option {
    value: T;
    label: string;
    hint?: string;
  }
  let { label, options, value, onchange }: { label: string; options: Option[]; value: T; onchange: (v: T) => void } = $props();
</script>

<div class="field">
  <span class="label">{label}</span>
  <div class="segmented" role="radiogroup" aria-label={label}>
    {#each options as o (o.value)}
      <button
        type="button"
        role="radio"
        aria-checked={o.value === value}
        class:selected={o.value === value}
        onclick={() => onchange(o.value)}
      >
        {o.label}
        {#if o.hint}<span class="hint">{o.hint}</span>{/if}
      </button>
    {/each}
  </div>
</div>

<style>
  .field {
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-width: 0;
  }
  .label {
    font-size: 13px;
    color: var(--muted);
  }
  .segmented {
    display: flex;
    background: var(--surface-2);
    border-radius: 9px;
    padding: 3px;
    gap: 3px;
  }
  button {
    flex: 1;
    border: none;
    background: none;
    border-radius: 7px;
    padding: 6px 10px;
    white-space: nowrap;
    display: flex;
    flex-direction: column;
    align-items: center;
    line-height: 1.25;
  }
  button.selected {
    background: var(--surface);
    box-shadow: 0 1px 2px rgb(0 0 0 / 0.15);
    font-weight: 600;
  }
  .hint {
    font-size: 11px;
    color: var(--muted);
    font-weight: 400;
  }
</style>
