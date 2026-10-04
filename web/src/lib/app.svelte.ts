// The app's state and actions: settings, model loading, recording, takes and playback.
import { applyGain, normalizationGainDb, SR, speechDbfs } from "./audio/level";
import { decodeFile, Player, resample } from "./audio/playback";
import { microphones, Recorder } from "./audio/recorder";
import { resolve, type BackendChoice, type Capabilities } from "./engine/backend";
import { Engine } from "./engine/client";
import type { ModelRef, Phase, Setup } from "./engine/protocol";
import {
  defaultDecoder, defaultEncoder, fileUrl, parseManifest,
  type Backend, type DecoderEntry, type EncoderEntry, type Manifest, type ModelEntry,
} from "./models/manifest";
import { clearDownloads, pruneDownloads, storedBytes } from "./models/download";

const MODELS_URL = new URL(import.meta.env.VITE_MODELS_URL ?? "models/", document.baseURI).href;
const WASM_PATHS = new URL("ort/", document.baseURI).href;
const MIN_SECONDS = 0.3;
const PREFS_KEY = "wesper-settings";

export interface Settings {
  encoder: string;
  /** The decoder, i.e. the voice the conversion speaks with. */
  decoder: string;
  backend: BackendChoice;
  /** Also convert each take with the other encoder(s). */
  compare: boolean;
  /** Also convert each take with the other voice(s). */
  compareVoices: boolean;
  microphone: string;
}

export interface Output {
  key: string;
  encoder: EncoderEntry;
  decoder: DecoderEntry;
  backend: Backend;
  status: "queued" | "running" | "done" | "error";
  /** Gain applied to the input for this encoder, or null if it takes the input as-is. */
  gainDb: number | null;
  /** At decoder.sampleRate. */
  samples: Float32Array | null;
  encodeMs: number;
  decodeMs: number;
  error: string;
}

export interface Take {
  id: number;
  name: string;
  /** 16 kHz, as the models receive it before level normalization. */
  samples: Float32Array;
  levelDbfs: number;
  outputs: Output[];
}

export interface Loading {
  name: string;
  phase: Phase;
  loaded: number;
  total: number;
}

function loadPrefs(): Partial<Settings> {
  try {
    return JSON.parse(localStorage.getItem(PREFS_KEY) ?? "{}");
  } catch {
    return {};
  }
}

const ref = (entry: ModelEntry): ModelRef => ({
  url: fileUrl(MODELS_URL, entry.file),
  bytes: entry.file.bytes,
  name: entry.file.path,
});

export class App {
  manifest = $state<Manifest | null>(null);
  caps = $state<Capabilities | null>(null);
  /** The app can't run: no models, no worker. */
  fatal = $state("");
  /** The last thing that went wrong, shown until dismissed. */
  error = $state("");
  notice = $state("");
  settings = $state<Settings>({ encoder: "sv", decoder: "sv-narrator", backend: "auto", compare: true, compareVoices: false, microphone: "" });
  loading = $state<Loading | null>(null);
  ready = $state(false);
  takes = $state<Take[]>([]);
  recording = $state(false);
  level = $state(-120);
  elapsed = $state(0);
  playing = $state<string | null>(null);
  devices = $state<MediaDeviceInfo[]>([]);
  stored = $state<number | null>(null);

  resolved = $derived(this.caps ? resolve(this.settings.backend, this.caps) : null);
  encoder = $derived(this.manifest?.encoders.find((e) => e.id === this.settings.encoder) ?? null);
  /** Encoders each take is converted with, the selected one first. */
  runEncoders = $derived.by(() => {
    const m = this.manifest;
    if (!m || !this.encoder) return [];
    const selected = this.encoder;
    return [selected, ...(this.settings.compare ? m.encoders.filter((e) => e.id !== selected.id) : [])];
  });
  decoder = $derived(this.manifest?.decoders.find((d) => d.id === this.settings.decoder) ?? null);
  /** Voices each take is converted with, the selected one first. */
  runDecoders = $derived.by(() => {
    const m = this.manifest;
    if (!m || !this.decoder) return [];
    const selected = this.decoder;
    return [selected, ...(this.settings.compareVoices ? m.decoders.filter((d) => d.id !== selected.id) : [])];
  });
  busy = $derived(this.takes.some((t) => t.outputs.some((o) => o.status === "queued" || o.status === "running")));

  private engine = new Engine();
  private ctx?: AudioContext;
  private recorder?: Recorder;
  private player?: Player;
  private pressed = false;
  private recordTimer?: ReturnType<typeof setInterval>;
  private nextTake = 1;
  private prepareRun = 0;

  async init() {
    // On GitHub Pages a first visit reloads once, to get WASM threads (index.html). Wait for that
    // rather than start downloads it would cut off.
    const sw = navigator.serviceWorker;
    if (!crossOriginIsolated && sw && !sw.controller) await Promise.race([sw.ready, new Promise((r) => setTimeout(r, 3000))]);
    try {
      const manifestUrl = new URL("models.json", MODELS_URL);
      const res = await fetch(manifestUrl).catch((e) => {
        const crossOrigin = manifestUrl.origin !== location.origin;
        throw new Error(`Couldn't load ${manifestUrl} (${e instanceof Error ? e.message : e}).` +
          (crossOrigin ? ` If the file exists, its server must allow this site (${location.origin}) with CORS.` : ""));
      });
      if (!res.ok) throw new Error(`No models at ${MODELS_URL} (HTTP ${res.status}). Export them with: .venv/bin/python web/export_models.py`);
      this.manifest = parseManifest(await res.json());
      this.caps = await this.engine.capabilities();
    } catch (e) {
      this.fatal = e instanceof Error ? e.message : String(e);
      return;
    }
    const prefs = loadPrefs();
    const m = this.manifest;
    this.settings = {
      encoder: m.encoders.some((e) => e.id === prefs.encoder) ? prefs.encoder! : defaultEncoder(m),
      decoder: m.decoders.some((d) => d.id === prefs.decoder) ? prefs.decoder! : defaultDecoder(m),
      backend: (["auto", "webgpu", "wasm"] as const).includes(prefs.backend!) ? prefs.backend! : "auto",
      compare: prefs.compare ?? true,
      compareVoices: prefs.compareVoices ?? false,
      microphone: prefs.microphone ?? "",
    };
    // Downloads of files no longer in models.json (earlier exports) only take up space.
    void pruneDownloads([...m.encoders, ...m.decoders].map((e) => ref(e).url)).then(async () => (this.stored = await storedBytes()));
    void this.refreshDevices();
    navigator.mediaDevices?.addEventListener?.("devicechange", () => void this.refreshDevices());
    void this.prepare();
    // With access already allowed, open the microphone now: opening it on the first press
    // takes long enough to cut off the start of the first take.
    try {
      if ((await navigator.permissions.query({ name: "microphone" as PermissionName })).state === "granted") await this.openMic();
    } catch {
      // can't tell (e.g. Firefox), or can't open: the first press opens it
    }
  }

  /** Changes settings, then loads whatever models they need. */
  update(change: Partial<Settings>) {
    Object.assign(this.settings, change);
    try {
      localStorage.setItem(PREFS_KEY, JSON.stringify(this.settings));
    } catch {
      // settings just aren't remembered
    }
    if ("microphone" in change) this.recorder?.close();
    if (["encoder", "decoder", "backend", "compare", "compareVoices"].some((k) => k in change)) void this.prepare();
  }

  private setup(): Setup {
    const r = this.resolved!;
    return { backend: r.backend, wasmPaths: WASM_PATHS };
  }

  private onProgress = (name: string, phase: Phase, loaded: number, total: number) => {
    this.loading = { name, phase, loaded, total };
  };

  async prepare() {
    const m = this.manifest;
    if (!m || !this.resolved) return;
    const run = ++this.prepareRun;
    this.ready = false;
    try {
      const { notCached } = await this.engine.prepare(this.setup(), this.runEncoders.map(ref), this.runDecoders.map(ref), this.onProgress);
      if (run === this.prepareRun) this.ready = true;
      if (notCached.length)
        this.notice = `This browser didn't let the page keep ${notCached.join(", ")} (a private window, or low on disk space?), so ${notCached.length > 1 ? "they download" : "it downloads"} again next visit.`;
    } catch (e) {
      if (run === this.prepareRun) this.error = `Couldn't load the models: ${e instanceof Error ? e.message : e}`;
    } finally {
      if (run === this.prepareRun) this.loading = null;
      this.stored = await storedBytes();
    }
  }

  async clearDownloads() {
    await clearDownloads();
    this.stored = await storedBytes();
    this.notice = "Downloaded models cleared. Loaded models stay in memory until you reload the page.";
  }

  private audio(): AudioContext {
    this.ctx ??= new AudioContext();
    void this.ctx.resume();
    this.player ??= new Player(this.ctx, (key) => (this.playing = key));
    return this.ctx;
  }

  private async openMic(): Promise<Recorder> {
    this.recorder ??= new Recorder(this.audio(), (db) => (this.level = db));
    await this.recorder.open(this.settings.microphone);
    void this.refreshDevices(); // labels appear once access is allowed
    return this.recorder;
  }

  async refreshDevices() {
    try {
      this.devices = await microphones();
    } catch {
      this.devices = [];
    }
  }

  // ------------------------------------------------------------ recording

  async pressStart() {
    if (this.pressed || this.fatal) return;
    this.pressed = true;
    this.error = this.notice = "";
    this.player?.stop();
    let recorder: Recorder;
    try {
      recorder = await this.openMic();
    } catch (e) {
      this.pressed = false;
      this.error = `Can't use the microphone: ${e instanceof Error ? e.message : e}`;
      return;
    }
    if (!this.pressed) return; // released while the microphone was opening
    recorder.start();
    this.recording = true;
    this.level = -120;
    const started = performance.now();
    const max = this.manifest?.maxSeconds ?? 120;
    this.recordTimer = setInterval(() => {
      this.elapsed = (performance.now() - started) / 1000;
      if (this.elapsed >= max) void this.pressEnd();
    }, 100);
  }

  async pressEnd() {
    this.pressed = false;
    if (!this.recording || !this.recorder) return;
    this.recording = false;
    clearInterval(this.recordTimer);
    try {
      const rec = await this.recorder.stop();
      await this.addTake("Microphone", await resample(rec.samples, rec.sampleRate, SR));
    } catch (e) {
      this.error = `Recording failed: ${e instanceof Error ? e.message : e}`;
    }
  }

  async addFile(file: File) {
    this.error = this.notice = "";
    this.audio();
    try {
      await this.addTake(file.name, await decodeFile(file, SR));
    } catch (e) {
      this.error = `Can't read ${file.name}: ${e instanceof Error ? e.message : e}`;
    }
  }

  private async addTake(name: string, samples: Float32Array) {
    const max = (this.manifest?.maxSeconds ?? 120) * SR;
    if (samples.length < MIN_SECONDS * SR) {
      this.notice = `That was too short (${(samples.length / SR).toFixed(2)} s). Hold the button while you whisper.`;
      return;
    }
    if (samples.length > max) {
      samples = samples.slice(0, max);
      this.notice = `Only the first ${max / SR} s are converted.`;
    }
    this.takes.unshift({ id: this.nextTake++, name, samples, levelDbfs: speechDbfs(samples), outputs: [] });
    await this.convert(this.takes[0]);
  }

  // ------------------------------------------------------------ conversion

  /** Converts a take with the current settings, adding one output per encoder and voice. Plays the first. */
  async convert(take: Take) {
    if (!this.manifest || !this.resolved) return;
    const setup = this.setup();
    const outputs = this.runEncoders.flatMap((encoder) => this.runDecoders.map((decoder): Output => ({
      key: `${take.id}:${take.outputs.length}:${encoder.id}:${decoder.id}`,
      encoder, decoder, backend: setup.backend, status: "queued",
      gainDb: encoder.targetDbfs === null ? null : normalizationGainDb(take.samples, encoder.targetDbfs, encoder.maxGainDb!),
      samples: null, encodeMs: 0, decodeMs: 0, error: "",
    })));
    const first = take.outputs.length;
    take.outputs.push(...outputs);
    for (let i = first; i < take.outputs.length; i++) {
      const out = take.outputs[i]; // the reactive copy
      out.status = "running";
      try {
        const input = out.gainDb === null ? take.samples : applyGain(take.samples, out.gainDb);
        const r = await this.engine.convert(setup, ref(out.encoder), ref(out.decoder), input, this.onProgress);
        this.loading = null;
        Object.assign(out, { samples: r.wav, encodeMs: r.encodeMs, decodeMs: r.decodeMs, status: "done" });
        if (i === first && take === this.takes[0]) this.play(out.key, r.wav, out.decoder.sampleRate);
      } catch (e) {
        this.loading = null;
        Object.assign(out, { status: "error", error: e instanceof Error ? e.message : String(e) });
      }
    }
  }

  removeTake(take: Take) {
    this.takes = this.takes.filter((t) => t.id !== take.id);
  }

  // ------------------------------------------------------------ playback

  play(key: string, samples: Float32Array, rate = SR) {
    this.audio();
    if (this.playing === key) this.player!.stop();
    else this.player!.play(key, samples, rate);
  }

  stop() {
    this.player?.stop();
  }

  /** Plays row n of the newest take: 0 is the input, 1.. its outputs. */
  playRow(n: number) {
    const take = this.takes[0];
    if (!take) return;
    if (n === 0) return this.play(`${take.id}:input`, take.samples);
    const out = take.outputs[n - 1];
    if (out?.samples) this.play(out.key, out.samples, out.decoder.sampleRate);
  }
}
