import { expect, test } from "bun:test";
import { decodeWav, encodeWav } from "./wav";

test("round trip is exact to half a 16-bit step", () => {
  const x = Float32Array.from({ length: 1000 }, (_, i) => 0.9 * Math.sin(i / 7));
  const { samples, sampleRate } = decodeWav(encodeWav(x, 16000));
  expect(sampleRate).toBe(16000);
  expect(samples.length).toBe(x.length);
  for (let i = 0; i < x.length; i++) expect(Math.abs(samples[i] - x[i])).toBeLessThanOrEqual(0.5 / 32768 + 1e-9);
});

test("header describes 16-bit PCM mono", () => {
  const bytes = encodeWav(new Float32Array(10), 16000);
  const v = new DataView(bytes.buffer);
  expect(bytes.length).toBe(44 + 20);
  expect(new TextDecoder().decode(bytes.subarray(0, 4))).toBe("RIFF");
  expect(v.getUint32(4, true)).toBe(bytes.length - 8);
  expect([v.getUint16(20, true), v.getUint16(22, true), v.getUint32(24, true), v.getUint16(34, true)]).toEqual([1, 1, 16000, 16]);
  expect(v.getUint32(40, true)).toBe(20);
});

test("clips instead of wrapping around", () => {
  const { samples } = decodeWav(encodeWav(Float32Array.of(1.7, -3, 1, -1), 16000));
  expect(Array.from(samples)).toEqual([32767 / 32768, -1, 32767 / 32768, -1]);
});

test("rejects what it can't read", () => {
  expect(() => decodeWav(new Uint8Array(44))).toThrow("not a WAV file");
});
