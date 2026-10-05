// Streaming: the vocoder runs on chunks of the mel, each with context on both sides, and the chunks
// play as they're ready. The vocoders are convolutional, so with enough context the chunks join into
// exactly the audio of one run over the whole mel. Same windows as stream_windows() in
// web/export_models.py, which checks that for every exported vocoder.

/** How a vocoder streams, in its mel frames (models.json). */
export interface StreamPlan {
  /** The first chunk: short, so playback starts soon. */
  firstFrames: number;
  /** The others: long, so the context costs little. */
  chunkFrames: number;
  /** Mel frames of context on each side of a chunk. */
  contextFrames: number;
}

/** One vocoder run: it plays mel frames [a, b), and runs on frames [s, e). */
export interface Window {
  a: number;
  b: number;
  s: number;
  e: number;
}

/** The vocoder runs for a mel of `frames` frames. The first runs on [0, first + context); the others on
 *  chunk + 2 * context frames, shifted back at the end, so that on WebGPU their shaders compile once. */
export function streamWindows(frames: number, p: StreamPlan): Window[] {
  const out: Window[] = [];
  const k = p.contextFrames;
  for (let a = 0; a < frames; ) {
    const b = Math.min(frames, a + (a === 0 ? p.firstFrames : p.chunkFrames));
    const e = Math.min(frames, b + k);
    const s = a === 0 ? 0 : Math.max(0, Math.min(a - k, e - (p.chunkFrames + 2 * k)));
    out.push({ a, b, s, e });
    a = b;
  }
  return out;
}

/** The window sizes, in mel frames, a long take runs the vocoder on: to warm their shaders up ahead. */
export function windowSizes(p: StreamPlan): number[] {
  return [p.firstFrames + p.contextFrames, p.chunkFrames + 2 * p.contextFrames];
}

/** Mel frames [s, e) of a mel [1, nMels, frames] (row-major), as a new [1, nMels, e - s] array. */
export function melWindow(mel: Float32Array, nMels: number, frames: number, s: number, e: number): Float32Array {
  const n = e - s;
  const out = new Float32Array(nMels * n);
  for (let m = 0; m < nMels; m++) out.set(mel.subarray(m * frames + s, m * frames + e), m * n);
  return out;
}

/** Window w's chunk out of the vocoder's output for it. The last chunk keeps the output's end, like
 *  HiFi-GAN's 8 extra samples, so the chunks add up to one run's output exactly. */
export function chunkSamples(out: Float32Array, w: Window, frames: number, hop: number): Float32Array {
  const lo = (w.a - w.s) * hop;
  return w.b === frames ? out.subarray(lo) : out.subarray(lo, (w.b - w.s) * hop);
}

/** How long after the first chunk is ready to start playing it, so that playback doesn't catch up with
 *  the conversion. The later windows are expected to take as long per mel frame as the first did
 *  (firstMs), with some slack. With a fast backend that's just the margin. */
export function startDelayMs(windows: Window[], firstMs: number, hop: number, sampleRate: number, marginMs = 20, slack = 1.25): number {
  if (windows.length < 2) return marginMs;
  const msPerFrame = (firstMs / (windows[0].e - windows[0].s)) * slack;
  const audioMs = (w: Window) => ((w.b - w.a) * hop * 1000) / sampleRate;
  let ready = 0; // when window i is ready, after the first was
  let played = audioMs(windows[0]); // how much plays before window i is needed
  let need = 0;
  for (let i = 1; i < windows.length; i++) {
    ready += msPerFrame * (windows[i].e - windows[i].s);
    need = Math.max(need, ready - played);
    played += audioMs(windows[i]);
  }
  return marginMs + need;
}
