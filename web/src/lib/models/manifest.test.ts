import { describe, expect, test } from "bun:test";
import { defaultEncoder, fileUrl, parseManifest, type Manifest } from "./manifest";

const file = (path: string) => ({ path, bytes: 100, sha256: "ab".repeat(32) });
const entry = (id: string) => ({ id, label: id, description: "", files: { fp32: file(`${id}.fp32.onnx`), fp16: file(`${id}.fp16.onnx`) } });
const manifest = (): Manifest => ({
  version: 1, sampleRate: 16000, hop: 320, maxSeconds: 120,
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
  test("rejects a missing precision", () => {
    const m = manifest();
    delete (m.encoders[1].files as Partial<Manifest["encoders"][0]["files"]>).fp16;
    expect(() => parseManifest(m)).toThrow("original has no valid fp16 file");
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
