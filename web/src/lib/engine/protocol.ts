// Messages between the page and the inference worker (worker.ts).
import type { Backend } from "../models/manifest";
import type { Capabilities } from "./backend";

/** A model file to run, as the worker needs it. */
export interface ModelRef {
  url: string;
  bytes: number;
  /** Short name for progress messages, e.g. "encoder-sv.onnx". */
  name: string;
}

/** Where the models run. */
export interface Setup {
  backend: Backend;
  wasmPaths: string;
}

export type Request =
  | { id: number; type: "capabilities" }
  /** Download, initialize and warm up models ahead of use; releases models of other setups. */
  | { id: number; type: "prepare"; setup: Setup; encoders: ModelRef[]; decoders: ModelRef[] }
  | { id: number; type: "convert"; setup: Setup; encoder: ModelRef; decoder: ModelRef; wav: Float32Array };

export interface PrepareResult {
  /** Models that couldn't be stored in the browser, so they download again next visit. */
  notCached: string[];
}

export interface ConvertResult {
  wav: Float32Array;
  encodeMs: number;
  decodeMs: number;
}

export type Phase = "download" | "cached" | "initialize" | "warm up";

export type Response =
  | { id: number; type: "progress"; name: string; phase: Phase; loaded: number; total: number }
  | { id: number; type: "done"; result: Capabilities | PrepareResult | ConvertResult }
  | { id: number; type: "error"; message: string };
