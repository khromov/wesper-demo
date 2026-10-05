// Messages between the page and the inference worker (worker.ts).
import type { Backend } from "../models/manifest";
import type { Capabilities } from "./backend";
import type { StreamPlan } from "./stream";

/** A model file to run, as the worker needs it. */
export interface ModelRef {
  url: string;
  bytes: number;
  /** Short name for progress messages, e.g. "encoder-sv.onnx". */
  name: string;
}

/** A voice: its decoder, and its vocoder if that's a file of its own (models.json version 3). Without
 *  one, the decoder file makes the audio itself (version 2) and can't stream. */
export interface VoiceRef {
  decoder: ModelRef;
  vocoder: ModelRef | null;
  plan: StreamPlan | null;
  /** The audio's sample rate, and its samples per mel frame. */
  sampleRate: number;
  hop: number;
}

/** Where the models run. */
export interface Setup {
  backend: Backend;
  wasmPaths: string;
}

export type Request =
  | { id: number; type: "capabilities" }
  /** Download, initialize and warm up models ahead of use (with stream, also for streaming); releases models of other setups. */
  | { id: number; type: "prepare"; setup: Setup; encoders: ModelRef[]; voices: VoiceRef[]; stream: boolean }
  /** With stream, the audio also comes in chunks as it's made ("chunk" messages), before "done". */
  | { id: number; type: "convert"; setup: Setup; encoder: ModelRef; voice: VoiceRef; wav: Float32Array; stream: boolean };

export interface PrepareResult {
  /** Models that couldn't be stored in the browser, so they download again next visit. */
  notCached: string[];
}

export interface ConvertResult {
  wav: Float32Array;
  encodeMs: number;
  decodeMs: number;
  /** Streamed: when the first chunk was ready, from the start. */
  firstMs: number | null;
}

/** A streamed chunk of the audio. The chunks follow each other without gaps or overlap. */
export interface Chunk {
  samples: Float32Array;
  index: number;
  count: number;
  /** For the first chunk: how long to wait before playing it, so that the rest stays ahead of playback. */
  startAfterMs: number;
}

export type Phase = "download" | "cached" | "initialize" | "warm up";

export type Response =
  | { id: number; type: "progress"; name: string; phase: Phase; loaded: number; total: number }
  | ({ id: number; type: "chunk" } & Chunk)
  | { id: number; type: "done"; result: Capabilities | PrepareResult | ConvertResult }
  | { id: number; type: "error"; message: string };
