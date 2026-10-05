// models.json, written by web/export_models.py: which models exist, their files, how each
// encoder's input level is set, and the sample rate each voice speaks at. Version 3 has each voice
// as FastSpeech2 (decoders) plus a vocoder file of its own (vocoders), which the app can stream;
// in version 2 a decoder file is both, and doesn't stream.
import type { StreamPlan } from "../engine/stream";

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

export interface VocoderEntry extends ModelEntry {
  /** Which vocoder (vocoders.py): "hifigan16k", "bigvgan22k". */
  vocoder: string;
  sampleRate: number;
  hop: number;
  stream: StreamPlan;
}

export interface DecoderEntry extends ModelEntry {
  /** The language the voice speaks: "sv", "en", ... */
  language: string;
  /** The voice's vocoder in vocoders (version 3), or null if this file makes the audio itself (version 2). */
  vocoderId: string | null;
  /** The vocoder in the decoder's graph (vocoders.py): "hifigan16k", "bigvgan22k", ... */
  vocoder: string;
  /** The output audio's sample rate, and its samples per mel frame. */
  sampleRate: number;
  hop: number;
  /** The vocoder is the run's own, fine-tuned on this decoder's output (decoder/finetune_vocoder.py). */
  vocoderFineTuned: boolean;
  /** The decoder was trained for HiFi-GAN: its mel is converted to the vocoder's on the way (bigvgan_preview.py). */
  melMap: boolean;
}

export interface Manifest {
  version: 2 | 3;
  /** The encoders' input: 16 kHz, one unit per 320 samples. */
  sampleRate: number;
  hop: number;
  maxSeconds: number;
  encoders: EncoderEntry[];
  /** Empty in version 2. */
  vocoders: VocoderEntry[];
  decoders: DecoderEntry[];
}

/** What decoders without vocoder settings use: models.json from before BigVGAN voices had none. */
const HIFIGAN16K = { vocoder: "hifigan16k", sampleRate: 16000, hop: 320 };

/** Names for the languages and vocoders the voices are chosen by. */
export const LANGUAGE_LABELS: Record<string, string> = { sv: "Swedish", en: "English" };
export const VOCODER_LABELS: Record<string, string> = { hifigan16k: "HiFi-GAN 16 kHz", bigvgan22k: "BigVGAN 22 kHz" };
export const languageLabel = (language: string) => LANGUAGE_LABELS[language] ?? language;
export const vocoderLabel = (vocoder: string) => VOCODER_LABELS[vocoder] ?? vocoder;

/** The encoder selected by default: the Swedish one when it was exported. */
export const DEFAULT_ENCODER = "sv";
/** The voice (decoder) selected by default: the Swedish narrator when it was exported. */
export const DEFAULT_DECODER = "sv-narrator";

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
  if (!m || (m.version !== 2 && m.version !== 3)) fail(`unsupported version ${m?.version}`);
  if (m.sampleRate !== 16000 || m.hop !== 320) fail(`expected 16 kHz audio and 320-sample frames`);
  if (!(typeof m.maxSeconds === "number" && m.maxSeconds > 0)) fail("maxSeconds is missing");
  if (!m.encoders?.length) fail("no encoders");
  if (!m.decoders?.length) fail("no decoders");
  m.encoders.forEach((e, i) => {
    checkEntry(e, `encoder ${i}`);
    if ((e.targetDbfs === null) !== (e.maxGainDb === null)) fail(`encoder ${e.id} needs both targetDbfs and maxGainDb, or neither`);
  });
  const positive = (x: unknown) => Number.isInteger(x) && (x as number) > 0;
  if (m.version === 3 && !m.vocoders?.length) fail("no vocoders");
  m.vocoders = m.version === 2 ? [] : m.vocoders!.map((v, i) => {
    checkEntry(v, `vocoder ${i}`);
    const p = v.stream;
    if (typeof v.vocoder !== "string" || !positive(v.sampleRate) || !positive(v.hop))
      fail(`vocoder ${v.id} needs a vocoder, a sampleRate and a hop`);
    if (!p || !positive(p.firstFrames) || !positive(p.chunkFrames) || !(Number.isInteger(p.contextFrames) && p.contextFrames >= 0))
      fail(`vocoder ${v.id} has no stream plan`);
    return { ...v, description: v.description ?? "" };
  });
  const vocoders = m.vocoders;
  m.decoders = m.decoders.map((d, i) => {
    checkEntry(d, `decoder ${i}`);
    // models.json from before voices had a language: WESPER's English one, or the Swedish narrator
    const o = d as Partial<DecoderEntry>;
    const v = { ...HIFIGAN16K, ...d, language: o.language ?? (d.id.startsWith("googletts") ? "en" : "sv"),
                vocoderId: m.version === 3 ? (o.vocoderId ?? "") : null,
                vocoderFineTuned: o.vocoderFineTuned === true, melMap: o.melMap === true };
    if (typeof v.vocoder !== "string" || !positive(v.sampleRate) || !positive(v.hop))
      fail(`decoder ${d.id} needs a vocoder, a sampleRate and a hop`);
    if (typeof v.language !== "string" || !v.language) fail(`decoder ${d.id} has no language`);
    if (v.vocoderId !== null) {
      const voc = vocoders.find((x) => x.id === v.vocoderId);
      if (!voc) fail(`decoder ${d.id}'s vocoder ${v.vocoderId || "(none)"} isn't in vocoders`);
      if (voc.vocoder !== v.vocoder || voc.sampleRate !== v.sampleRate || voc.hop !== v.hop)
        fail(`decoder ${d.id} doesn't match its vocoder ${voc.id}`);
    }
    return v;
  });
  return m as Manifest;
}

/** URL of a model file. The hash in the query string makes a re-exported file a new cache entry. */
export function fileUrl(base: string, file: ModelFile): string {
  return new URL(`${file.path}?v=${file.sha256.slice(0, 16)}`, base).href;
}

export function defaultEncoder(m: Manifest): string {
  return m.encoders.some((e) => e.id === DEFAULT_ENCODER) ? DEFAULT_ENCODER : m.encoders[0].id;
}

export function defaultDecoder(m: Manifest): string {
  return m.decoders.some((d) => d.id === DEFAULT_DECODER) ? DEFAULT_DECODER : m.decoders[0].id;
}

/** The voices' languages, in models.json order. */
export function languages(m: Manifest): string[] {
  return [...new Set(m.decoders.map((d) => d.language))];
}

/** The voice to switch to for another language: the same vocoder if it has one, else its first voice. */
export function voiceForLanguage(m: Manifest, language: string, vocoder: string): string {
  const voices = m.decoders.filter((d) => d.language === language);
  return (voices.find((d) => d.vocoder === vocoder) ?? voices[0]).id;
}
