import { describe, expect, test } from "bun:test";
import { defaultEncoder, fileUrl, parseManifest, type Manifest } from "./manifest";

const file = (path: string) => ({ path, bytes: 100, sha256: "ab".repeat(32) });
const entry = (id: string) => ({ id, label: id, description: "", file: file(`${id}.onnx`) });
const manifest = (): Manifest => ({
  version: 2, sampleRate: 16000, hop: 320, maxSeconds: 120,
  encoders: [{ ...entry("sv"), targetDbfs: -20, maxGainDb: 40 }, { ...entry("original"), targetDbfs: null, maxGainDb: null }],
  decoders: [entry("googletts")],
});

describe("parseManifest", () => {
  test("accepts what export_models.py writes", () => {
    expect(parseManifest(manifest()).encoders.map((e) => e.id)).toEqual(["sv", "original"]);
  });
  test("rejects other audio formats", () => {
    expect(() => parseManifest({ ...manifest(), sampleRate: 22050 })).toThrow("16 kHz");
  });
  test("rejects a missing or broken file", () => {
    const m = manifest();
    m.encoders[1].file.sha256 = "not a hash";
    expect(() => parseManifest(m)).toThrow("original has no valid file");
  });
  test("rejects the fp16/fp32 layout of version 1", () => {
    expect(() => parseManifest({ ...manifest(), version: 1 })).toThrow("unsupported version 1");
  });
  test("rejects half a level setting", () => {
    const m = manifest();
    m.encoders[0].maxGainDb = null;
    expect(() => parseManifest(m)).toThrow("both targetDbfs and maxGainDb");
  });
  test("says how to fix it", () => {
    expect(() => parseManifest({})).toThrow("web/export_models.py");
  });
});

test("fileUrl resolves against the base and adds a version", () => {
  expect(fileUrl("http://x/app/models/", file("e.onnx"))).toBe(`http://x/app/models/e.onnx?v=${"ab".repeat(8)}`);
});

test("the Swedish encoder is the default when present", () => {
  expect(defaultEncoder(manifest())).toBe("sv");
  const m = manifest();
  m.encoders.reverse();
  expect(defaultEncoder(m)).toBe("sv");
  m.encoders = m.encoders.filter((e) => e.id !== "sv");
  expect(defaultEncoder(m)).toBe("original");
});
