// Messages between the page and the inference worker (worker.ts).
import type { Backend, Precision } from "../models/manifest";
import type { Capabilities } from "./backend";

/** A model file to run, as the worker needs it. */
export interface ModelRef {
  url: string;
  bytes: number;
  /** Short name for progress messages, e.g. "encoder-sv.fp16.onnx". */
  name: string;
}

/** A set of models that run together: they share a backend and a precision. */
export interface Setup {
  backend: Backend;
  precision: Precision;
  wasmPaths: string;
}

export type Request =
  | { id: number; type: "capabilities" }
  /** Download, initialize and warm up models ahead of use; releases models of other setups. */
  | { id: number; type: "prepare"; setup: Setup; encoders: ModelRef[]; decoder: ModelRef }
  | { id: number; type: "convert"; setup: Setup; encoder: ModelRef; decoder: ModelRef; wav: Float32Array };

export interface ConvertResult {
  wav: Float32Array;
  encodeMs: number;
  decodeMs: number;
}

export type Phase = "download" | "cached" | "initialize" | "warm up";

export type Response =
  | { id: number; type: "progress"; name: string; phase: Phase; loaded: number; total: number }
  | { id: number; type: "done"; result: Capabilities | ConvertResult | null }
  | { id: number; type: "error"; message: string };
