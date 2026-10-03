/// <reference lib="webworker" />
// Runs the models with ONNX Runtime Web, off the page's main thread. Requests run one at a time,
// in order. Sessions stay loaded for the current backend and precision; switching either
// releases the others, so GPU memory holds one setup at a time.
import * as ort from "onnxruntime-web";
import { download } from "../models/download";
import type { Capabilities } from "./backend";
import type { ConvertResult, ModelRef, Phase, Request, Response, Setup } from "./protocol";

declare const self: DedicatedWorkerGlobalScope;

const sessions = new Map<string, Promise<ort.InferenceSession>>(); // key(): setup + url
const warmed = new Set<string>();

const post = (msg: Response, transfer: Transferable[] = []) => self.postMessage(msg, transfer);
const progress = (id: number, name: string, phase: Phase, loaded = 0, total = 1) =>
  post({ id, type: "progress", name, phase, loaded, total });
const setupKey = (s: Setup) => `${s.backend}|${s.precision}|`;
const key = (s: Setup, m: ModelRef) => setupKey(s) + m.url;
const threads = () => (self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1);

async function capabilities(): Promise<Capabilities> {
  const caps: Capabilities = { webgpu: false, shaderF16: false, adapter: "", threads: threads() };
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (adapter) {
      caps.webgpu = true;
      caps.shaderF16 = adapter.features.has("shader-f16");
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
      const bytes = await download(model.url, model.bytes, (loaded, total, fromCache) => {
        const now = performance.now();
        if (loaded === total || now - last > 100) {
          last = now;
          progress(id, model.name, fromCache ? "cached" : "download", loaded, total);
        }
      });
      progress(id, model.name, "initialize");
      return ort.InferenceSession.create(bytes, {
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

async function run(enc: ort.InferenceSession, dec: ort.InferenceSession, wav: Float32Array): Promise<ConvertResult> {
  const t0 = performance.now();
  const { units } = await enc.run({ wav: new ort.Tensor("float32", wav, [1, wav.length]) });
  const t1 = performance.now();
  const out = await dec.run({ units });
  const t2 = performance.now();
  const result = { wav: new Float32Array(out.wav.data as Float32Array), encodeMs: t1 - t0, decodeMs: t2 - t1 };
  units.dispose();
  out.wav.dispose();
  return result;
}

/** Sessions for an encoder and the decoder. The first run compiles GPU shaders (seconds), so a
 *  new pair runs once on a second of silence before it's used. */
async function pair(setup: Setup, encoder: ModelRef, decoder: ModelRef, id: number) {
  const [enc, dec] = [await session(setup, encoder, id), await session(setup, decoder, id)];
  const [ke, kd] = [key(setup, encoder), key(setup, decoder)];
  if (!warmed.has(ke) || !warmed.has(kd)) {
    progress(id, encoder.name, "warm up");
    await run(enc, dec, new Float32Array(16000));
    warmed.add(ke).add(kd);
  }
  return [enc, dec] as const;
}

async function handle(req: Request) {
  try {
    if (req.type === "capabilities") {
      post({ id: req.id, type: "done", result: await capabilities() });
    } else if (req.type === "prepare") {
      await useSetup(req.setup);
      for (const encoder of req.encoders) await pair(req.setup, encoder, req.decoder, req.id);
      post({ id: req.id, type: "done", result: null });
    } else {
      await useSetup(req.setup);
      const [enc, dec] = await pair(req.setup, req.encoder, req.decoder, req.id);
      const result = await run(enc, dec, req.wav);
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
