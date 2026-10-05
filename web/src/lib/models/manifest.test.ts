import { describe, expect, test } from "bun:test";
import { defaultDecoder, defaultEncoder, fileUrl, languages, parseManifest, voiceForLanguage, type Manifest } from "./manifest";

const file = (path: string) => ({ path, bytes: 100, sha256: "ab".repeat(32) });
const entry = (id: string) => ({ id, label: id, description: "", file: file(`${id}.onnx`) });
const manifest = (): Manifest => ({
  version: 2, sampleRate: 16000, hop: 320, maxSeconds: 120, vocoders: [],
  encoders: [{ ...entry("sv"), targetDbfs: -20, maxGainDb: 40 }, { ...entry("original"), targetDbfs: null, maxGainDb: null }],
  decoders: [entry("sv-narrator"), entry("sv-narrator-bigvgan"), entry("googletts"), entry("googletts-bigvgan")].map((d) => ({
    ...d, language: d.id.startsWith("googletts") ? "en" : "sv", vocoderId: null, vocoderFineTuned: d.id === "sv-narrator-bigvgan", melMap: d.id === "googletts-bigvgan",
    ...(d.id.endsWith("bigvgan") ? { vocoder: "bigvgan22k", sampleRate: 22050, hop: 256 } : { vocoder: "hifigan16k", sampleRate: 16000, hop: 320 }),
  })),
});

describe("parseManifest", () => {
  test("accepts what export_models.py writes", () => {
    expect(parseManifest(manifest()).encoders.map((e) => e.id)).toEqual(["sv", "original"]);
  });
  test("rejects other audio formats", () => {
    expect(() => parseManifest({ ...manifest(), sampleRate: 22050 })).toThrow("16 kHz");
  });
  test("keeps each voice's sample rate", () => {
    expect(parseManifest(manifest()).decoders.map((d) => [d.id, d.vocoder, d.sampleRate, d.hop])).toEqual([
      ["sv-narrator", "hifigan16k", 16000, 320],
      ["sv-narrator-bigvgan", "bigvgan22k", 22050, 256],
      ["googletts", "hifigan16k", 16000, 320],
      ["googletts-bigvgan", "bigvgan22k", 22050, 256],
    ]);
  });
  test("keeps each voice's language", () => {
    expect(parseManifest(manifest()).decoders.map((d) => d.language)).toEqual(["sv", "sv", "en", "en"]);
  });
  test("keeps whether a voice's vocoder is fine-tuned or its mel converted", () => {
    expect(parseManifest(manifest()).decoders.map((d) => [d.id, d.vocoderFineTuned, d.melMap])).toEqual([
      ["sv-narrator", false, false], ["sv-narrator-bigvgan", true, false], ["googletts", false, false], ["googletts-bigvgan", false, true],
    ]);
  });
  test("voices from before those flags have neither", () => {
    const m = manifest();
    for (const d of m.decoders) {
      delete (d as Partial<typeof d>).vocoderFineTuned;
      delete (d as Partial<typeof d>).melMap;
    }
    expect(parseManifest(m).decoders.every((d) => d.vocoderFineTuned === false && d.melMap === false)).toBe(true);
  });
  test("voices from before languages: googletts is English, the narrator Swedish", () => {
    const m = manifest();
    for (const d of m.decoders) delete (d as Partial<typeof d>).language;
    expect(parseManifest(m).decoders.map((d) => d.language)).toEqual(["sv", "sv", "en", "en"]);
  });
  test("voices without vocoder settings are HiFi-GAN 16 kHz, as before BigVGAN", () => {
    const m = { ...manifest(), decoders: [entry("googletts")] };
    expect(parseManifest(m).decoders[0]).toMatchObject({ vocoder: "hifigan16k", sampleRate: 16000, hop: 320 });
  });
  test("rejects a broken voice sample rate", () => {
    const m = manifest();
    m.decoders[1].sampleRate = 0;
    expect(() => parseManifest(m)).toThrow("sv-narrator-bigvgan needs a vocoder, a sampleRate and a hop");
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

// Version 3: each voice is FastSpeech2 (decoders) plus a vocoder file (vocoders), as export_models.py writes it.
const plan = (first: number, chunk: number, context: number) => ({ firstFrames: first, chunkFrames: chunk, contextFrames: context });
const v3 = (): Manifest => {
  const m = manifest();
  m.version = 3;
  m.vocoders = [
    { ...entry("hifigan16k"), vocoder: "hifigan16k", sampleRate: 16000, hop: 320, stream: plan(25, 100, 20) },
    { ...entry("bigvgan22k"), vocoder: "bigvgan22k", sampleRate: 22050, hop: 256, stream: plan(44, 173, 35) },
    { ...entry("sv-narrator-bigvgan"), vocoder: "bigvgan22k", sampleRate: 22050, hop: 256, stream: plan(44, 173, 35) },
  ];
  for (const d of m.decoders) d.vocoderId = d.id === "sv-narrator-bigvgan" ? d.id : d.vocoder;
  return m;
};

describe("version 3", () => {
  test("gives each voice its vocoder file and how to stream it", () => {
    const m = parseManifest(v3());
    expect(m.decoders.map((d) => d.vocoderId)).toEqual(["hifigan16k", "sv-narrator-bigvgan", "hifigan16k", "bigvgan22k"]);
    expect(m.vocoders[0].stream).toEqual(plan(25, 100, 20));
  });
  test("version 2 voices have no vocoder file of their own", () => {
    const m = parseManifest(manifest());
    expect(m.vocoders).toEqual([]);
    expect(m.decoders.every((d) => d.vocoderId === null)).toBe(true);
  });
  test("rejects a voice without its vocoder", () => {
    const m = v3();
    m.vocoders = m.vocoders.filter((v) => v.id !== "bigvgan22k");
    expect(() => parseManifest(m)).toThrow("googletts-bigvgan's vocoder bigvgan22k isn't in vocoders");
  });
  test("rejects a voice that doesn't match its vocoder", () => {
    const m = v3();
    m.decoders[0].vocoderId = "bigvgan22k";
    expect(() => parseManifest(m)).toThrow("sv-narrator doesn't match its vocoder bigvgan22k");
  });
  test("rejects a vocoder without a stream plan", () => {
    const m = v3();
    (m.vocoders[0] as Partial<(typeof m.vocoders)[0]>).stream = undefined;
    expect(() => parseManifest(m)).toThrow("hifigan16k has no stream plan");
  });
  test("needs vocoders", () => {
    expect(() => parseManifest({ ...v3(), vocoders: [] })).toThrow("no vocoders");
  });
});

describe("choosing a voice by language", () => {
  test("lists the languages in models.json order", () => {
    expect(languages(manifest())).toEqual(["sv", "en"]);
  });
  test("keeps the vocoder when the other language has it", () => {
    expect(voiceForLanguage(manifest(), "en", "bigvgan22k")).toBe("googletts-bigvgan");
    expect(voiceForLanguage(manifest(), "sv", "hifigan16k")).toBe("sv-narrator");
  });
  test("else takes the language's first voice", () => {
    const m = manifest();
    m.decoders = m.decoders.filter((d) => d.id !== "googletts-bigvgan");
    expect(voiceForLanguage(m, "en", "bigvgan22k")).toBe("googletts");
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

test("the Swedish narrator is the default voice when present", () => {
  expect(defaultDecoder(manifest())).toBe("sv-narrator");
  const m = manifest();
  m.decoders.reverse();
  expect(defaultDecoder(m)).toBe("sv-narrator");
  m.decoders = m.decoders.filter((d) => d.id !== "sv-narrator");
  expect(defaultDecoder(m)).toBe("googletts-bigvgan"); // otherwise the first one
});
