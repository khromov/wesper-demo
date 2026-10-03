// Resampling, audio file decoding and playback, using the browser's own audio engine.

/** Mono audio at `rate`, resampled by an OfflineAudioContext. */
export async function resample(samples: Float32Array, fromRate: number, rate: number): Promise<Float32Array> {
  if (fromRate === rate || samples.length === 0) return samples;
  const ctx = new OfflineAudioContext(1, Math.max(1, Math.round((samples.length * rate) / fromRate)), rate);
  const buf = ctx.createBuffer(1, samples.length, fromRate);
  buf.getChannelData(0).set(samples);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.start();
  return (await ctx.startRendering()).getChannelData(0);
}

/** Any audio file the browser can decode, as mono at `rate`. */
export async function decodeFile(file: Blob, rate: number): Promise<Float32Array> {
  const decoded = await new OfflineAudioContext(1, 1, rate).decodeAudioData(await file.arrayBuffer());
  const mono = new Float32Array(decoded.length);
  for (let c = 0; c < decoded.numberOfChannels; c++) {
    const ch = decoded.getChannelData(c);
    for (let i = 0; i < mono.length; i++) mono[i] += ch[i] / decoded.numberOfChannels;
  }
  return decoded.sampleRate === rate ? mono : resample(mono, decoded.sampleRate, rate);
}

/** Plays one clip at a time; starting another stops the current one. */
export class Player {
  private current?: AudioBufferSourceNode;
  /** Key of the clip playing, or null. */
  playing: string | null = null;

  constructor(private ctx: AudioContext, private onChange: (playing: string | null) => void = () => {}) {}

  play(key: string, samples: Float32Array, rate: number) {
    this.stop();
    const buf = this.ctx.createBuffer(1, samples.length, rate);
    buf.getChannelData(0).set(samples);
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.ctx.destination);
    src.onended = () => {
      if (this.current !== src) return;
      this.current = undefined;
      this.set(null);
    };
    this.current = src;
    void this.ctx.resume();
    src.start();
    this.set(key);
  }

  stop() {
    const src = this.current;
    this.current = undefined;
    src?.stop();
    this.set(null);
  }

  private set(key: string | null) {
    this.playing = key;
    this.onChange(key);
  }
}
