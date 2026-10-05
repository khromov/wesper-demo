import { describe, expect, test } from "bun:test";
import { chunkSamples, melWindow, startDelayMs, streamWindows, windowSizes, type StreamPlan } from "./stream";

// HiFi-GAN 16 kHz's plan in models.json (export_models.stream_plan): 0.5 s, 2 s, 0.4 s in 20 ms frames.
const PLAN: StreamPlan = { firstFrames: 25, chunkFrames: 100, contextFrames: 20 };

describe("streamWindows", () => {
  test("plays every frame once, in order, with context on both sides", () => {
    for (const frames of [1, 24, 25, 26, 44, 45, 125, 140, 141, 300, 1001]) {
      const w = streamWindows(frames, PLAN);
      expect(w.map((x) => x.a)).toEqual([0, ...w.slice(0, -1).map((x) => x.b)]);
      expect(w.at(-1)!.b).toBe(frames);
      for (const { a, b, s, e } of w) {
        expect(0 <= s && s <= a && a < b && b <= e && e <= frames).toBe(true);
        expect(a - s).toBeGreaterThanOrEqual(Math.min(20, a));
        expect(e - b).toBeGreaterThanOrEqual(Math.min(20, frames - b));
      }
    }
  });
  test("a long take runs the vocoder on two window sizes only", () => {
    const sizes = new Set(streamWindows(1001, PLAN).map((w) => w.e - w.s));
    expect([...sizes].sort()).toEqual(windowSizes(PLAN).sort());
  });
  test("matches export_models.stream_windows", () => {
    // stream_windows(300, plan) in Python
    expect(streamWindows(300, PLAN).map((w) => [w.a, w.b, w.s, w.e])).toEqual([
      [0, 25, 0, 45], [25, 125, 5, 145], [125, 225, 105, 245], [225, 300, 160, 300],
    ]);
  });
});

describe("chunks", () => {
  // A "vocoder" whose output depends only on each frame (hop samples of its value), plus HiFi-GAN's 8 extra
  // samples at the end: the chunks must join into its output for the whole mel exactly.
  const hop = 4;
  const nMels = 3;
  const vocoder = (mel: Float32Array, frames: number) => {
    const out = new Float32Array(frames * hop + 8);
    for (let t = 0; t < frames; t++) out.fill(mel[t] + 2 * mel[frames + t], t * hop, (t + 1) * hop);
    return out;
  };
  test("melWindow cuts every band", () => {
    const mel = Float32Array.from({ length: nMels * 10 }, (_, i) => i);
    expect([...melWindow(mel, nMels, 10, 3, 6)]).toEqual([3, 4, 5, 13, 14, 15, 23, 24, 25]);
  });
  test("join into the whole mel's audio", () => {
    const frames = 300;
    const mel = Float32Array.from({ length: nMels * frames }, () => Math.random());
    const full = vocoder(mel, frames);
    const parts = streamWindows(frames, PLAN).map((w) =>
      chunkSamples(vocoder(melWindow(mel, nMels, frames, w.s, w.e), w.e - w.s), w, frames, hop));
    const joined = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
    parts.reduce((at, p) => (joined.set(p, at), at + p.length), 0);
    expect(joined).toEqual(full);
  });
});

describe("startDelayMs", () => {
  const windows = streamWindows(500, PLAN); // 10 s of 20 ms frames
  test("fast enough: just the margin", () => {
    // the first window (45 frames, 0.9 s of mel) in 50 ms: ~18x real time
    expect(startDelayMs(windows, 50, 320, 16000)).toBe(20);
  });
  test("slower than real time: waits so that the conversion stays ahead", () => {
    // 2 ms of compute per 1 ms of audio: playing 10 s needs the last chunk ~10 s after the first
    const delay = startDelayMs(windows, 45 * 20 * 2, 320, 16000);
    expect(delay).toBeGreaterThan(10_000);
    // check: with that head start, each window is ready before playback reaches it
    const perFrame = ((45 * 20 * 2) / 45) * 1.25;
    let ready = 0;
    let played = (25 * 320 * 1000) / 16000 + delay;
    for (const w of windows.slice(1)) {
      ready += perFrame * (w.e - w.s);
      expect(ready).toBeLessThanOrEqual(played);
      played += ((w.b - w.a) * 320 * 1000) / 16000;
    }
  });
  test("one chunk: nothing to wait for", () => {
    expect(startDelayMs(streamWindows(20, PLAN), 5000, 320, 16000)).toBe(20);
  });
});
