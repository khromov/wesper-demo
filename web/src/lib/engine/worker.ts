/// <reference lib="webworker" />
// Runs the models with ONNX Runtime Web, off the page's main thread. Requests run one at a time,
// in order. Sessions stay loaded for the current backend; switching it releases the others.
import * as ort from "onnxruntime-web";
import { download } from "../models/download";
import type { Capabilities } from "./backend";
import type { Chunk, ConvertResult, ModelRef, Phase, Request, Response, Setup, VoiceRef } from "./protocol";
import { chunkSamples, melWindow, startDelayMs, streamWindows, windowSizes } from "./stream";

declare const self: DedicatedWorkerGlobalScope;

const sessions = new Map<string, Promise<ort.InferenceSession>>(); // key(): setup + url
const warmed = new Set<string>();
const melBands = new Map<string, number>(); // vocoder key -> its mel's bands, once seen
const notCached = new Set<string>(); // names of models the browser wouldn't store

const post = (msg: Response, transfer: Transferable[] = []) => self.postMessage(msg, transfer);
const progress = (id: number, name: string, phase: Phase, loaded = 0, total = 1) =>
  post({ id, type: "progress", name, phase, loaded, total });
const setupKey = (s: Setup) => `${s.backend}|`;
const key = (s: Setup, m: ModelRef) => setupKey(s) + m.url;
const threads = () => (self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1);

async function capabilities(): Promise<Capabilities> {
  const caps: Capabilities = { webgpu: false, adapter: "", threads: threads() };
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (adapter) {
      caps.webgpu = true;
      caps.adapter = [adapter.info?.vendor, adapter.info?.architecture].filter(Boolean).join(" ");
    }
  } catch {
    // no usable GPU
  }
  return caps;
}

async function useSetup(setup: Setup) {
  // ORT reads these when it starts, on the first session; they're the same for every setup.
  ort.env.wasm.wasmPaths = setup.wasmPaths;
  ort.env.wasm.numThreads = threads();
  ort.env.logLevel = "error";
  for (const [k, s] of sessions) {
    if (k.startsWith(setupKey(setup))) continue;
    sessions.delete(k);
    warmed.delete(k);
    await s.then((x) => x.release()).catch(() => {});
  }
}

/** The model's session, downloaded and initialized on first use. */
function session(setup: Setup, model: ModelRef, id: number): Promise<ort.InferenceSession> {
  const k = key(setup, model);
  let s = sessions.get(k);
  if (!s) {
    s = (async () => {
      let last = 0;
      const { data, cached } = await download(model.url, model.bytes, (loaded, total, fromCache) => {
        const now = performance.now();
        if (loaded === total || now - last > 100) {
          last = now;
          progress(id, model.name, fromCache ? "cached" : "download", loaded, total);
        }
      });
      if (!cached) notCached.add(model.name);
      progress(id, model.name, "initialize");
      return ort.InferenceSession.create(data, {
        executionProviders: [setup.backend],
        graphOptimizationLevel: "all",
        logSeverityLevel: 3, // errors only: it warns that shape ops run on the CPU, which is intended
      });
    })();
    sessions.set(k, s);
    s.catch(() => sessions.delete(k)); // a later request tries again
  }
  return s;
}

interface Loaded {
  enc: ort.InferenceSession;
  dec: ort.InferenceSession;
  /** The voice's vocoder (models.json version 3); null if the decoder makes the audio itself (version 2). */
  voc: ort.InferenceSession | null;
}

function concat(parts: Float32Array[]): Float32Array {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  parts.reduce((at, p) => (out.set(p, at), at + p.length), 0);
  return out;
}

/** Encoder, then decoder (units -> mel), then vocoder (mel -> audio). With onChunk and a vocoder of its
 *  own, the vocoder runs on chunks of the mel (stream.ts), each passed on as soon as it's made. */
async function run(m: Loaded, voice: VoiceRef, wav: Float32Array, onChunk?: (c: Chunk) => void): Promise<ConvertResult & { bands: number }> {
  const t0 = performance.now();
  const { units } = await m.enc.run({ wav: new ort.Tensor("float32", wav, [1, wav.length]) });
  const t1 = performance.now();
  const out = await m.dec.run({ units });
  units.dispose();
  if (!m.voc) {
    const result = new Float32Array(out.wav.data as Float32Array);
    out.wav.dispose();
    return { wav: result, encodeMs: t1 - t0, decodeMs: performance.now() - t1, firstMs: null, bands: 0 };
  }
  const mel = out.mel;
  const [, bands, frames] = mel.dims as number[];
  let result: Float32Array;
  let firstMs: number | null = null;
  if (!onChunk || !voice.plan) {
    const { wav: w } = await m.voc.run({ mel });
    result = new Float32Array(w.data as Float32Array);
    w.dispose();
  } else {
    const data = mel.data as Float32Array;
    const windows = streamWindows(frames, voice.plan);
    const parts: Float32Array[] = [];
    for (const [i, w] of windows.entries()) {
      const started = performance.now();
      const input = new ort.Tensor("float32", melWindow(data, bands, frames, w.s, w.e), [1, bands, w.e - w.s]);
      const { wav: o } = await m.voc.run({ mel: input });
      const samples = chunkSamples(o.data as Float32Array, w, frames, voice.hop).slice();
      input.dispose();
      o.dispose();
      let startAfterMs = 0;
      if (i === 0) {
        firstMs = performance.now() - t0;
        startAfterMs = startDelayMs(windows, performance.now() - started, voice.hop, voice.sampleRate);
      }
      parts.push(samples);
      onChunk({ samples, index: i, count: windows.length, startAfterMs });
    }
    result = concat(parts);
  }
  mel.dispose();
  return { wav: result, encodeMs: t1 - t0, decodeMs: performance.now() - t1, firstMs, bands };
}

/** Sessions for an encoder and a voice. The first run compiles GPU shaders (seconds), so a new set runs
 *  once on a second of silence before it's used; for streaming on WebGPU, the vocoder also runs on its
 *  window sizes, so that compiling them doesn't hold up the first stream. */
async function load(setup: Setup, encoder: ModelRef, voice: VoiceRef, id: number, stream: boolean): Promise<Loaded> {
  const m: Loaded = {
    enc: await session(setup, encoder, id),
    dec: await session(setup, voice.decoder, id),
    voc: voice.vocoder ? await session(setup, voice.vocoder, id) : null,
  };
  const keys = [encoder, voice.decoder, ...(voice.vocoder ? [voice.vocoder] : [])].map((r) => key(setup, r));
  if (keys.some((k) => !warmed.has(k))) {
    progress(id, encoder.name, "warm up");
    const { bands } = await run(m, voice, new Float32Array(16000));
    if (voice.vocoder) melBands.set(key(setup, voice.vocoder), bands);
    keys.forEach((k) => warmed.add(k));
  }
  if (stream && m.voc && voice.vocoder && voice.plan && setup.backend === "webgpu") {
    const k = key(setup, voice.vocoder);
    if (!warmed.has(k + "|stream")) {
      progress(id, voice.vocoder.name, "warm up");
      const bands = melBands.get(k) ?? (await run(m, voice, new Float32Array(16000))).bands;
      for (const n of windowSizes(voice.plan)) {
        const mel = new ort.Tensor("float32", new Float32Array(bands * n).fill(-11.5), [1, bands, n]); // silence
        (await m.voc.run({ mel })).wav.dispose();
        mel.dispose();
      }
      warmed.add(k + "|stream");
    }
  }
  return m;
}

async function handle(req: Request) {
  try {
    if (req.type === "capabilities") {
      post({ id: req.id, type: "done", result: await capabilities() });
    } else if (req.type === "prepare") {
      await useSetup(req.setup);
      for (const encoder of req.encoders) for (const voice of req.voices) await load(req.setup, encoder, voice, req.id, req.stream);
      post({ id: req.id, type: "done", result: { notCached: [...notCached] } });
    } else {
      await useSetup(req.setup);
      const m = await load(req.setup, req.encoder, req.voice, req.id, req.stream);
      const onChunk = req.stream ? (c: Chunk) => post({ id: req.id, type: "chunk", ...c }) : undefined;
      const { bands: _, ...result } = await run(m, req.voice, req.wav, onChunk);
      post({ id: req.id, type: "done", result }, [result.wav.buffer]);
    }
  } catch (e) {
    post({ id: req.id, type: "error", message: e instanceof Error ? e.message : String(e) });
  }
}

let queue = Promise.resolve();
self.onmessage = (e: MessageEvent<Request>) => {
  queue = queue.then(() => handle(e.data));
};
