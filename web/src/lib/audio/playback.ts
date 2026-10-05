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

/** A clip that plays while it's still being made: chunks play back to back as they come. */
export interface StreamHandle {
  /** Queues the next chunk. The first one plays startAfterMs after it arrives. */
  push(samples: Float32Array, startAfterMs?: number): void;
  /** No more chunks: playing ends after the last one. */
  end(): void;
}

/** Plays one clip at a time; starting another stops the current one. */
export class Player {
  private current?: AudioBufferSourceNode;
  private streaming?: { id: number; sources: AudioBufferSourceNode[] };
  private streams = 0;
  /** For streams: a context at the stream's own sample rate, by rate (see stream()). */
  private contexts = new Map<number, AudioContext>();
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

  /** Plays a clip as its chunks come (StreamHandle). Chunks are scheduled back to back, sample-exact. In a
   *  context at the clip's own rate: resampling each chunk to the device's rate on its own would leave a
   *  tiny step at every join. A chunk that comes too late plays as soon as it can, after a gap. */
  stream(key: string, rate: number): StreamHandle {
    this.stop();
    const id = ++this.streams;
    let ctx = this.contexts.get(rate);
    if (!ctx) {
      try {
        ctx = new AudioContext({ sampleRate: rate });
      } catch {
        ctx = this.ctx; // a browser without contexts at that rate: joins are resampled, but play
      }
      this.contexts.set(rate, ctx);
    }
    void ctx.resume();
    const sources: AudioBufferSourceNode[] = [];
    this.streaming = { id, sources };
    let next = 0;
    let open = true;
    let pending = 0;
    const finished = () => {
      if (open || pending > 0 || this.streaming?.id !== id) return;
      this.streaming = undefined;
      this.set(null);
    };
    return {
      push: (samples, startAfterMs = 0) => {
        if (this.streaming?.id !== id || samples.length === 0) return; // stopped, or another clip
        const buf = ctx.createBuffer(1, samples.length, rate);
        buf.getChannelData(0).set(samples);
        const src = ctx.createBufferSource();
        src.buffer = buf;
        src.connect(ctx.destination);
        const now = ctx.currentTime;
        const at = Math.max(sources.length ? next : now + startAfterMs / 1000, now + 0.005);
        src.start(at);
        next = at + buf.duration;
        pending++;
        src.onended = () => {
          pending--;
          finished();
        };
        sources.push(src);
        if (sources.length === 1) this.set(key);
      },
      end: () => {
        open = false;
        finished();
      },
    };
  }

  stop() {
    const src = this.current;
    this.current = undefined;
    src?.stop();
    const stream = this.streaming;
    this.streaming = undefined;
    for (const s of stream?.sources ?? []) {
      try {
        s.stop();
      } catch {
        // not started yet
      }
    }
    this.set(null);
  }

  private set(key: string | null) {
    this.playing = key;
    this.onChange(key);
  }
}
