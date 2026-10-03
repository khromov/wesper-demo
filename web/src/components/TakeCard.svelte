<script lang="ts">
  import type { Take } from "../lib/app.svelte";
  import { SR } from "../lib/audio/level";
  import { encodeWav } from "../lib/audio/wav";

  let {
    take,
    latest,
    playing,
    onplay,
    onconvert,
    onremove,
  }: {
    take: Take;
    latest: boolean;
    playing: string | null;
    onplay: (key: string, samples: Float32Array) => void;
    onconvert: () => void;
    onremove: () => void;
  } = $props();

  const seconds = $derived(take.samples.length / SR);
  const inputKey = $derived(`${take.id}:input`);
  const signed = (db: number) => `${db >= 0 ? "+" : "−"}${Math.abs(db).toFixed(1)} dB`;
  const backendName = { webgpu: "WebGPU", wasm: "WASM" };

  function download(samples: Float32Array, name: string) {
    const url = URL.createObjectURL(new Blob([encodeWav(samples, SR)], { type: "audio/wav" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: name });
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
  }
</script>

<article class="card take" data-take={take.id}>
  <header>
    <div>
      <strong>Take {take.id}</strong>
      <span class="muted small">· {take.name} · {seconds.toFixed(1)} s · speech level {take.levelDbfs.toFixed(1)} dBFS</span>
    </div>
    <div class="actions">
      <button class="button small" onclick={onconvert} title="Convert again with the current settings">Run again</button>
      <button class="button small" onclick={onremove} aria-label="Remove take {take.id}">Remove</button>
    </div>
  </header>

  <ol>
    <li class:playing={playing === inputKey} data-row="input">
      <button class="play" onclick={() => onplay(inputKey, take.samples)} aria-label="Play input">
        {playing === inputKey ? "■" : "▶"}
      </button>
      <kbd class:hidden={!latest}>0</kbd>
      <div class="what">
        <span>Input</span>
        <span class="muted small">16 kHz, as recorded</span>
      </div>
      <button class="dl" onclick={() => download(take.samples, `take${take.id}-input.wav`)} aria-label="Download input">⤓</button>
    </li>
    {#each take.outputs as out, i (out.key)}
      <li class:playing={playing === out.key} data-row="output" data-encoder={out.encoder.id} data-status={out.status}>
        <button class="play" disabled={!out.samples} onclick={() => out.samples && onplay(out.key, out.samples)} aria-label="Play {out.encoder.label}">
          {playing === out.key ? "■" : "▶"}
        </button>
        <kbd class:hidden={!latest}>{i + 1}</kbd>
        <div class="what">
          <span>
            {out.encoder.label}
            <span class="chip">{backendName[out.backend]}</span>
          </span>
          <span class="muted small">
            {#if out.status === "queued"}
              Waiting…
            {:else if out.status === "running"}
              Converting…
            {:else if out.status === "error"}
              <span class="error">{out.error}</span>
            {:else}
              {out.gainDb === null ? "input as-is" : `input ${signed(out.gainDb)} to ${out.encoder.targetDbfs} dBFS`}
              · {((out.encodeMs + out.decodeMs) / 1000).toFixed(2)} s
              <span title="encoder + decoder">({(out.encodeMs / 1000).toFixed(2)} + {(out.decodeMs / 1000).toFixed(2)})</span>
            {/if}
          </span>
        </div>
        {#if out.samples}
          {@const samples = out.samples}
          <button class="dl" onclick={() => download(samples, `take${take.id}-${out.encoder.id}-${out.backend}.wav`)} aria-label="Download {out.encoder.label}">⤓</button>
        {/if}
      </li>
    {/each}
  </ol>
</article>

<style>
  .take {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  header {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
    align-items: baseline;
  }
  .actions {
    display: flex;
    gap: 6px;
  }
  ol {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  li {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 8px;
    border-radius: 8px;
  }
  li.playing {
    background: var(--accent-soft);
  }
  .play {
    width: 36px;
    height: 36px;
    flex: none;
    border-radius: 50%;
    border: none;
    background: var(--accent);
    color: var(--accent-text);
    font-size: 13px;
  }
  .what {
    display: flex;
    flex-direction: column;
    min-width: 0;
    flex: 1;
  }
  .chip {
    font-size: 11px;
    font-family: var(--mono);
    background: var(--surface-2);
    border-radius: 4px;
    padding: 1px 5px;
    margin-left: 4px;
  }
  kbd {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--muted);
    border: 1px solid var(--border);
    border-bottom-width: 2px;
    border-radius: 4px;
    padding: 0 5px;
  }
  kbd.hidden {
    visibility: hidden; /* only the newest take has shortcuts; keeps the rows aligned */
  }
  .dl {
    border: none;
    background: none;
    color: var(--muted);
    font-size: 18px;
    padding: 4px 8px;
  }
  .error {
    color: var(--error-text);
  }
</style>
