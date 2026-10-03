import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import fixture from "./level.fixture.json";
import { applyGain, normalizationGainDb, SR, speechDbfs } from "./level";
import { decodeWav } from "./wav";

// Same signals as tests/test_web_level_fixture.py, which computes the fixture with Python's speech_dbfs().
function lcg(seed: number, n: number): Float64Array {
  const out = new Float64Array(n);
  let state = seed >>> 0;
  for (let i = 0; i < n; i++) {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    out[i] = (state / 2 ** 32) * 2 - 1;
  }
  return out;
}
const tone = (freq: number, amp: number, n: number) =>
  Float64Array.from({ length: n }, (_, i) => amp * Math.sin((2 * Math.PI * freq * i) / SR));
const concat = (...parts: Float64Array[]) => {
  const out = new Float64Array(parts.reduce((n, p) => n + p.length, 0));
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
};

function signals(): Record<string, Float32Array> {
  const click = tone(330, 0.05, 1.5 * SR);
  click[8000] = 0.9;
  const noise7 = lcg(7, 2 * SR);
  const modulated = Float64Array.from({ length: 2 * SR }, (_, i) => {
    const t = i / SR;
    return 0.2 * Math.sin(2 * Math.PI * 150 * t) * (0.5 + 0.5 * Math.sin(2 * Math.PI * 3 * t)) + 1e-4 * noise7[i];
  });
  const s: Record<string, Float64Array> = {
    silence: new Float64Array(SR),
    short: tone(220, 0.1, 100),
    tone_padded: concat(new Float64Array(SR / 2), tone(220, 0.1, SR), new Float64Array(SR / 2)),
    noise: lcg(1, SR).map((v) => 0.01 * v),
    tone_click: click,
    modulated,
    quiet_noise: lcg(3, SR).map((v) => 1e-5 * v),
  };
  const out: Record<string, Float32Array> = Object.fromEntries(Object.entries(s).map(([k, v]) => [k, Float32Array.from(v)]));
  out.sample_whisper = decodeWav(readFileSync(join(import.meta.dir, "../../../../sample_whisper.wav"))).samples;
  return out;
}

describe("speechDbfs matches colab/prepare_data.py", () => {
  const sigs = signals();
  test("the fixture and the test cover the same signals", () => {
    expect(Object.keys(sigs).sort()).toEqual(Object.keys(fixture.signals).sort());
  });
  for (const [name, want] of Object.entries(fixture.signals)) {
    test(name, () => {
      const x = sigs[name];
      expect(speechDbfs(x)).toBeCloseTo(want.speechDbfs, 6);
      expect(normalizationGainDb(x, fixture.targetDbfs, fixture.maxGainDb)).toBeCloseTo(want.gainDb, 6);
    });
  }
});

describe("applyGain", () => {
  test("reaches the target level", () => {
    const x = signals().noise;
    const y = applyGain(x, normalizationGainDb(x, -20, 40));
    expect(speechDbfs(y)).toBeCloseTo(-20, 3);
  });
  test("keeps peaks above 1.0 instead of clipping", () => {
    const y = applyGain(Float32Array.of(0.5, -0.5), 12);
    expect(y[0]).toBeGreaterThan(1.9);
    expect(y[1]).toBeLessThan(-1.9);
  });
  test("leaves the input untouched", () => {
    const x = Float32Array.of(0.1, 0.2);
    applyGain(x, 6);
    expect(Array.from(x)).toEqual([Math.fround(0.1), Math.fround(0.2)]);
  });
});
