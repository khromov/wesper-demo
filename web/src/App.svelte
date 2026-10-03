<script lang="ts">
  import { onMount } from "svelte";
  import PushToTalk from "./components/PushToTalk.svelte";
  import Segmented from "./components/Segmented.svelte";
  import TakeCard from "./components/TakeCard.svelte";
  import { App } from "./lib/app.svelte";
  import type { BackendChoice } from "./lib/engine/backend";
  import type { Precision } from "./lib/models/manifest";

  const app = new App();
  // For poking at the state from devtools, and for scripts/e2e.ts.
  (window as unknown as { __wesper: App }).__wesper = app;
  onMount(() => void app.init());

  let fileInput = $state<HTMLInputElement>();

  const mb = (bytes: number) => (bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.round(bytes / 1e6)} MB`);
  const backendName = { webgpu: "WebGPU", wasm: "WebAssembly" };
  const others = $derived(app.manifest?.encoders.filter((e) => e.id !== app.settings.encoder).map((e) => e.label) ?? []);

  const status = $derived.by(() => {
    const l = app.loading;
    if (!l) return null;
    if (l.phase === "download") return `Downloading ${l.name} · ${mb(l.loaded)} of ${mb(l.total)}`;
    if (l.phase === "cached") return `Loading ${l.name} from this browser's cache`;
    if (l.phase === "initialize") return `Initializing ${l.name}`;
    return app.resolved?.backend === "webgpu" ? `Warming up ${l.name} (compiling GPU shaders)` : `Warming up ${l.name}`;
  });

  function typing(e: KeyboardEvent) {
    return !!(e.target as HTMLElement | null)?.closest?.("input, select, textarea");
  }

  function keydown(e: KeyboardEvent) {
    if (typing(e) || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.code === "Space") {
      e.preventDefault();
      if (!e.repeat) void app.pressStart();
    } else if (/^Digit\d$/.test(e.code) && !e.repeat) {
      app.playRow(Number(e.code.slice(5)));
    } else if (e.key === "Escape") {
      app.stop();
    }
  }

  function keyup(e: KeyboardEvent) {
    if (e.code !== "Space" || typing(e)) return;
    e.preventDefault();
    void app.pressEnd();
  }

  function pickFile() {
    const file = fileInput?.files?.[0];
    if (fileInput) fileInput.value = "";
    if (file) void app.addFile(file);
  }
</script>

<svelte:window onkeydown={keydown} onkeyup={keyup} onblur={() => void app.pressEnd()} />

<main>
  <header class="top">
    <div>
      <h1>WESPER</h1>
      <p class="muted">Whispered speech to normal speech, converted in your browser. No audio leaves this device.</p>
    </div>
    {#if app.resolved && app.caps}
      <p class="engine small" data-testid="engine">
        <span class="dot" class:ready={app.ready}></span>
        {backendName[app.resolved.backend]} · {app.resolved.precision}
        {#if app.resolved.backend === "webgpu" && app.caps.adapter}<span class="muted">· {app.caps.adapter}</span>{/if}
        {#if app.resolved.backend === "wasm"}<span class="muted">· {app.caps.threads} threads</span>{/if}
      </p>
    {/if}
  </header>

  {#if app.fatal}
    <div class="banner error" role="alert"><span>{app.fatal}</span></div>
  {:else if app.manifest}
    <section class="card settings" aria-label="Settings">
      <div class="grid">
        <Segmented
          label="Encoder"
          value={app.settings.encoder}
          options={app.manifest.encoders.map((e) => ({ value: e.id, label: e.label }))}
          onchange={(encoder) => app.update({ encoder })}
        />
        <Segmented
          label="Precision"
          value={app.settings.precision}
          options={(["fp16", "fp32"] as Precision[]).map((p) => ({ value: p, label: p, hint: mb(app.downloadBytes(p)) }))}
          onchange={(precision) => app.update({ precision })}
        />
        <Segmented
          label="Backend"
          value={app.settings.backend}
          options={[
            { value: "auto" as BackendChoice, label: "Auto", hint: app.caps ? (app.caps.webgpu ? "WebGPU" : "WASM") : "" },
            { value: "webgpu" as BackendChoice, label: "WebGPU", hint: app.caps && !app.caps.webgpu ? "unavailable" : "" },
            { value: "wasm" as BackendChoice, label: "WASM", hint: app.caps ? `${app.caps.threads} threads` : "" },
          ]}
          onchange={(backend) => app.update({ backend })}
        />
      </div>
      {#if app.encoder}
        <p class="small muted desc">
          {app.encoder.description}.
          {app.encoder.targetDbfs === null ? "Input goes in at its recorded level." : `Input is normalized to ${app.encoder.targetDbfs} dBFS speech level, as in training.`}
        </p>
      {/if}
      <div class="row">
        {#if others.length}
          <label class="check">
            <input type="checkbox" checked={app.settings.compare} onchange={(e) => app.update({ compare: e.currentTarget.checked })} />
            Also convert with {others.join(", ")}, to compare
          </label>
        {/if}
        <label class="mic small">
          <span class="muted">Microphone</span>
          <select value={app.settings.microphone} onchange={(e) => app.update({ microphone: e.currentTarget.value })}>
            <option value="">System default</option>
            {#each app.devices.filter((d) => d.deviceId && d.deviceId !== "default") as d, i (d.deviceId)}
              <option value={d.deviceId}>{d.label || `Microphone ${i + 1}`}</option>
            {/each}
          </select>
        </label>
      </div>
      {#each app.resolved?.notes ?? [] as note (note)}
        <div class="banner warn small">{note}</div>
      {/each}
    </section>

    <div class="status small" aria-live="polite">
      {#if status}
        <span>{status}</span>
        {#if app.loading?.phase === "download"}
          <progress max={app.loading.total} value={app.loading.loaded}></progress>
        {/if}
      {:else if app.ready}
        <span class="muted">Models ready.</span>
      {:else if !app.error}
        <span class="muted">Loading models…</span>
      {/if}
    </div>

    <PushToTalk
      recording={app.recording}
      level={app.level}
      elapsed={app.elapsed}
      disabled={!!app.fatal}
      onstart={() => void app.pressStart()}
      onend={() => void app.pressEnd()}
    />
    <p class="small muted center">
      <button class="linkish" onclick={() => fileInput?.click()}>Convert an audio file instead</button>
      <input bind:this={fileInput} type="file" accept="audio/*" hidden onchange={pickFile} data-testid="file" />
      · <kbd>0</kbd>–<kbd>9</kbd> play the newest take's rows · <kbd>Esc</kbd> stops
    </p>

    {#if app.error}
      <div class="banner error small" role="alert"><span>{app.error}</span><button onclick={() => (app.error = "")} aria-label="Dismiss">✕</button></div>
    {/if}
    {#if app.notice}
      <div class="banner warn small" role="status"><span>{app.notice}</span><button onclick={() => (app.notice = "")} aria-label="Dismiss">✕</button></div>
    {/if}

    <section class="takes" aria-label="Takes">
      {#each app.takes as take, i (take.id)}
        <TakeCard
          {take}
          latest={i === 0}
          playing={app.playing}
          onplay={(key, samples) => app.play(key, samples)}
          onconvert={() => void app.convert(take)}
          onremove={() => app.removeTake(take)}
        />
      {:else}
        <p class="muted center small empty">Your takes appear here. Each one is converted with the encoders above, and the first result plays automatically.</p>
      {/each}
    </section>
  {:else}
    <p class="muted">Loading…</p>
  {/if}

  <footer class="small muted">
    <span>
      <a href="https://lab.rekimoto.org/projects/wesper/">WESPER</a> (Rekimoto, CHI 2023) on
      <a href="https://onnxruntime.ai/docs/tutorials/web/">ONNX Runtime Web</a>.
    </span>
    {#if app.stored}
      <span>{mb(app.stored)} stored in this browser · <button class="linkish" onclick={() => void app.clearDownloads()}>clear</button></span>
    {/if}
  </footer>
</main>

<style>
  main {
    max-width: 780px;
    margin: 0 auto;
    padding: 24px 16px 48px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    flex-wrap: wrap;
  }
  h1 {
    margin: 0;
    font-size: 26px;
    letter-spacing: 0.02em;
  }
  .top p {
    margin: 2px 0 0;
  }
  .engine {
    display: flex;
    align-items: center;
    gap: 6px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 4px 12px;
    white-space: nowrap;
  }
  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--muted);
  }
  .dot.ready {
    background: var(--ok);
  }
  .settings {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
  }
  .desc {
    margin: 0;
  }
  .row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }
  .check {
    display: flex;
    gap: 8px;
    align-items: center;
  }
  .mic {
    display: flex;
    gap: 8px;
    align-items: center;
  }
  .mic select {
    max-width: 260px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--surface);
    padding: 3px 6px;
  }
  .status {
    display: flex;
    align-items: center;
    gap: 10px;
    min-height: 20px;
  }
  progress {
    flex: 1;
    max-width: 240px;
    accent-color: var(--accent);
  }
  .center {
    text-align: center;
    margin: 0;
  }
  .linkish {
    border: none;
    background: none;
    padding: 0;
    color: var(--accent);
    text-decoration: underline;
  }
  kbd {
    font-family: var(--mono);
    font-size: 12px;
    border: 1px solid var(--border);
    border-bottom-width: 2px;
    border-radius: 4px;
    padding: 0 4px;
  }
  .takes {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .empty {
    padding: 24px 0;
  }
  footer {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    border-top: 1px solid var(--border);
    padding-top: 12px;
  }
  footer a {
    color: inherit;
  }
</style>
