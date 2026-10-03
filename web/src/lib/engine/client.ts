// The page's side of the inference worker: one promise per request, plus progress callbacks.
import type { Capabilities } from "./backend";
import type { ConvertResult, ModelRef, Phase, Request, Response, Setup } from "./protocol";

export type Progress = (name: string, phase: Phase, loaded: number, total: number) => void;
type Pending = { resolve: (v: unknown) => void; reject: (e: Error) => void; onProgress?: Progress };
type Body<T> = T extends unknown ? Omit<T, "id"> : never;

export class Engine {
  private worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  private pending = new Map<number, Pending>();
  private nextId = 1;

  constructor() {
    this.worker.onmessage = (e: MessageEvent<Response>) => {
      const msg = e.data;
      const p = this.pending.get(msg.id);
      if (!p) return;
      if (msg.type === "progress") return p.onProgress?.(msg.name, msg.phase, msg.loaded, msg.total);
      this.pending.delete(msg.id);
      if (msg.type === "done") p.resolve(msg.result);
      else p.reject(new Error(msg.message));
    };
    this.worker.onerror = (e) => {
      for (const p of this.pending.values()) p.reject(new Error(`inference worker failed: ${e.message}`));
      this.pending.clear();
    };
  }

  private call<T>(body: Body<Request>, onProgress?: Progress, transfer: Transferable[] = []): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, onProgress });
      this.worker.postMessage({ ...body, id }, transfer);
    });
  }

  capabilities(): Promise<Capabilities> {
    return this.call({ type: "capabilities" });
  }

  prepare(setup: Setup, encoders: ModelRef[], decoder: ModelRef, onProgress?: Progress): Promise<null> {
    return this.call({ type: "prepare", setup, encoders, decoder }, onProgress);
  }

  /** Converts 16 kHz audio. `wav` is copied, so the caller keeps its array. */
  convert(setup: Setup, encoder: ModelRef, decoder: ModelRef, wav: Float32Array, onProgress?: Progress): Promise<ConvertResult> {
    const copy = new Float32Array(wav);
    return this.call({ type: "convert", setup, encoder, decoder, wav: copy }, onProgress, [copy.buffer]);
  }
}
