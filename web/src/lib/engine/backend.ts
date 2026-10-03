// Picks where the models run (WebGPU or WebAssembly) and in which precision, from what the
// browser offers. The worker reports the capabilities; this logic stays pure so it can be tested.
import type { Backend, Precision } from "../models/manifest";

export type BackendChoice = "auto" | Backend;

export interface Capabilities {
  webgpu: boolean;
  /** The GPU can run fp16 shaders, which the fp16 models need on WebGPU. */
  shaderF16: boolean;
  /** e.g. "apple metal-3", or "" if the browser doesn't say. */
  adapter: string;
  /** WebAssembly threads; 1 without cross-origin isolation. */
  threads: number;
}

export interface Resolved {
  backend: Backend;
  precision: Precision;
  /** Why the result differs from what was asked for, if it does. */
  notes: string[];
}

export function resolve(choice: BackendChoice, precision: Precision, caps: Capabilities): Resolved {
  const notes: string[] = [];
  let backend: Backend = choice === "auto" ? (caps.webgpu ? "webgpu" : "wasm") : choice;
  if (backend === "webgpu" && !caps.webgpu) {
    backend = "wasm";
    notes.push("WebGPU isn't available in this browser, so the models run on WebAssembly.");
  }
  if (backend === "webgpu" && precision === "fp16" && !caps.shaderF16) {
    precision = "fp32";
    notes.push("This GPU can't run fp16 shaders, so the fp32 models are used.");
  }
  if (backend === "wasm" && caps.threads <= 1)
    notes.push("WebAssembly runs single-threaded here (the page isn't cross-origin isolated), so it's slow.");
  return { backend, precision, notes };
}
