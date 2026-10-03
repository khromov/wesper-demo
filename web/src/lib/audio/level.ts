// Speech-level normalization, ported from colab/prepare_data.py (speech_dbfs) and
// whisper_normal.py (normalize_level). The Swedish encoder was trained on input normalized this
// way, so its input must be too. level.fixture.json pins this port to the Python results.

export const SR = 16000;
export const HOP = 320; // 20 ms frames

/** numpy.percentile with its default linear interpolation, on an ascending-sorted array. */
function percentile(sorted: Float64Array, q: number): number {
  const pos = (q / 100) * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, sorted.length - 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** Speech level in dBFS: the mean power of the 20 ms frames that aren't silence (more than 30 dB
 *  below the loudest), counting frames within 20 dB of their 90th percentile, each capped there. */
export function speechDbfs(x: Float32Array): number {
  const n = Math.floor(x.length / HOP);
  if (n === 0) return -120;
  const energy = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = i * HOP; j < (i + 1) * HOP; j++) s += x[j] * x[j];
    energy[i] = s / HOP;
  }
  const floor = percentile(energy.slice().sort(), 99) / 1000;
  const speech = energy.filter((e) => e >= floor).sort();
  const ref = percentile(speech, 90);
  let sum = 0;
  let count = 0;
  for (const e of speech) {
    if (e >= ref / 100) {
      sum += Math.min(e, ref);
      count++;
    }
  }
  return 10 * Math.log10(sum / count + 1e-12);
}

/** Gain in dB that brings x to targetDbfs speech level, at most maxGainDb. */
export function normalizationGainDb(x: Float32Array, targetDbfs: number, maxGainDb: number): number {
  return Math.min(targetDbfs - speechDbfs(x), maxGainDb);
}

/** x scaled by gainDb. Peaks above 1.0 are kept, not clipped, as in normalize_level(). */
export function applyGain(x: Float32Array, gainDb: number): Float32Array {
  const g = Math.fround(10 ** (gainDb / 20));
  const y = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) y[i] = x[i] * g;
  return y;
}
