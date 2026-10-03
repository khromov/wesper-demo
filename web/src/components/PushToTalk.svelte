<script lang="ts">
  let {
    recording,
    level,
    elapsed,
    disabled,
    onstart,
    onend,
  }: { recording: boolean; level: number; elapsed: number; disabled: boolean; onstart: () => void; onend: () => void } = $props();

  // -80 dBFS (quiet room) .. 0 dBFS, as a share of the meter
  const meter = $derived(Math.max(0, Math.min(1, (level + 80) / 80)));

  function down(e: PointerEvent) {
    if (e.button !== 0 || disabled) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    onstart();
  }
</script>

<button
  type="button"
  class="ptt"
  class:recording
  {disabled}
  onpointerdown={down}
  onpointerup={onend}
  onpointercancel={onend}
  oncontextmenu={(e) => e.preventDefault()}
  aria-pressed={recording}
>
  {#if recording}
    <span class="title">Recording… {elapsed.toFixed(1)} s</span>
    <span class="meter" aria-hidden="true"><span style:width="{meter * 100}%"></span></span>
    <span class="sub">{level > -120 ? `${level.toFixed(0)} dBFS` : " "}</span>
  {:else}
    <span class="title">Hold to whisper</span>
    <span class="sub">or hold <kbd>Space</kbd>, then release to convert</span>
  {/if}
</button>

<style>
  .ptt {
    width: 100%;
    min-height: 132px;
    border: 2px solid var(--accent);
    background: var(--accent-soft);
    border-radius: var(--radius);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 16px;
    user-select: none;
    -webkit-user-select: none;
    touch-action: none;
  }
  .ptt.recording {
    border-color: var(--record);
    background: var(--record-soft);
  }
  .title {
    font-size: 24px;
    font-weight: 650;
  }
  .sub {
    color: var(--muted);
    font-size: 13px;
    min-height: 1.2em;
  }
  kbd {
    font-family: var(--mono);
    font-size: 12px;
    border: 1px solid var(--border);
    border-bottom-width: 2px;
    border-radius: 4px;
    padding: 0 4px;
    background: var(--surface);
  }
  .meter {
    width: min(320px, 80%);
    height: 8px;
    border-radius: 4px;
    background: var(--surface);
    overflow: hidden;
  }
  .meter span {
    display: block;
    height: 100%;
    background: var(--record);
    transition: width 80ms linear;
  }
</style>
