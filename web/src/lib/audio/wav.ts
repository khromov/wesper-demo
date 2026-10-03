// 16-bit PCM mono WAV encoding (for downloads) and decoding (for the tests).

/** A 16-bit PCM mono WAV file, scaled by 32768 like soundfile. Samples are clipped to [-1, 1). */
export function encodeWav(samples: Float32Array, sampleRate: number): Uint8Array<ArrayBuffer> {
  const bytes = new Uint8Array(44 + samples.length * 2);
  const v = new DataView(bytes.buffer);
  const text = (off: number, s: string) => [...s].forEach((c, i) => v.setUint8(off + i, c.charCodeAt(0)));
  text(0, "RIFF");
  v.setUint32(4, 36 + samples.length * 2, true);
  text(8, "WAVE");
  text(12, "fmt ");
  v.setUint32(16, 16, true); // fmt chunk size
  v.setUint16(20, 1, true); // PCM
  v.setUint16(22, 1, true); // mono
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true); // byte rate
  v.setUint16(32, 2, true); // block align
  v.setUint16(34, 16, true); // bits per sample
  text(36, "data");
  v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    v.setInt16(44 + i * 2, Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32768))), true);
  }
  return bytes;
}

/** Reads a 16-bit PCM WAV, averaging channels to mono, scaled like soundfile (int / 32768). */
export function decodeWav(data: Uint8Array): { samples: Float32Array; sampleRate: number } {
  const v = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const tag = (off: number) => String.fromCharCode(...data.subarray(off, off + 4));
  if (tag(0) !== "RIFF" || tag(8) !== "WAVE") throw new Error("not a WAV file");
  let channels = 0;
  let sampleRate = 0;
  for (let off = 12; off + 8 <= data.length; off += 8 + v.getUint32(off + 4, true) + (v.getUint32(off + 4, true) & 1)) {
    const size = v.getUint32(off + 4, true);
    if (tag(off) === "fmt ") {
      if (v.getUint16(off + 8, true) !== 1 || v.getUint16(off + 22, true) !== 16) throw new Error("only 16-bit PCM WAV is supported");
      channels = v.getUint16(off + 10, true);
      sampleRate = v.getUint32(off + 12, true);
    } else if (tag(off) === "data") {
      if (!channels) throw new Error("WAV data before its fmt chunk");
      const n = Math.floor(size / 2 / channels);
      const samples = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let c = 0; c < channels; c++) s += v.getInt16(off + 8 + (i * channels + c) * 2, true);
        samples[i] = s / channels / 32768;
      }
      return { samples, sampleRate };
    }
  }
  throw new Error("WAV file has no data chunk");
}
