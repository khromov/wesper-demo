// models.json, written by web/export_models.py: which models exist, their files, and how each
// encoder's input level is set.

export type Backend = "webgpu" | "wasm";

export interface ModelFile {
  path: string;
  bytes: number;
  sha256: string;
}

export interface ModelEntry {
  id: string;
  label: string;
  description: string;
  file: ModelFile;
}

export interface EncoderEntry extends ModelEntry {
  /** Speech level the encoder's input is normalized to, or null to pass input through unchanged. */
  targetDbfs: number | null;
  maxGainDb: number | null;
}

export interface Manifest {
  version: 2;
  sampleRate: number;
  hop: number;
  maxSeconds: number;
  encoders: EncoderEntry[];
  decoders: ModelEntry[];
}

/** The encoder selected by default: the Swedish one when it was exported. */
export const DEFAULT_ENCODER = "sv";

function fail(msg: string): never {
  throw new Error(`models.json: ${msg}. Re-run .venv/bin/python web/export_models.py`);
}

function checkEntry(e: unknown, where: string): asserts e is ModelEntry {
  const o = e as Partial<ModelEntry>;
  if (!o || typeof o.id !== "string" || typeof o.label !== "string") fail(`${where} needs an id and a label`);
  const f = o.file;
  if (!f || typeof f.path !== "string" || !(f.bytes > 0) || !/^[0-9a-f]{64}$/.test(f.sha256 ?? ""))
    fail(`${where} ${o.id} has no valid file`);
}

export function parseManifest(json: unknown): Manifest {
  const m = json as Partial<Manifest>;
  if (!m || m.version !== 2) fail(`unsupported version ${m?.version}`);
  if (m.sampleRate !== 16000 || m.hop !== 320) fail(`expected 16 kHz audio and 320-sample frames`);
  if (!(typeof m.maxSeconds === "number" && m.maxSeconds > 0)) fail("maxSeconds is missing");
  if (!m.encoders?.length) fail("no encoders");
  if (!m.decoders?.length) fail("no decoders");
  m.encoders.forEach((e, i) => {
    checkEntry(e, `encoder ${i}`);
    if ((e.targetDbfs === null) !== (e.maxGainDb === null)) fail(`encoder ${e.id} needs both targetDbfs and maxGainDb, or neither`);
  });
  m.decoders.forEach((d, i) => checkEntry(d, `decoder ${i}`));
  return m as Manifest;
}

/** URL of a model file. The hash in the query string makes a re-exported file a new cache entry. */
export function fileUrl(base: string, file: ModelFile): string {
  return new URL(`${file.path}?v=${file.sha256.slice(0, 16)}`, base).href;
}

export function defaultEncoder(m: Manifest): string {
  return m.encoders.some((e) => e.id === DEFAULT_ENCODER) ? DEFAULT_ENCODER : m.encoders[0].id;
}
