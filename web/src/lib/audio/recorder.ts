// Push-to-talk microphone capture. The microphone opens on first use and stays open, so later
// recordings start instantly; samples are kept only between start() and stop().

// The browser's voice processing is off: noise suppression treats whispers as noise, and
// automatic gain would change the level the encoders are sensitive to.
const CONSTRAINTS: MediaTrackConstraints = {
  channelCount: 1,
  echoCancellation: false,
  noiseSuppression: false,
  autoGainControl: false,
};

// Posts mono blocks of 2048 samples, and whatever is left when asked to flush.
const WORKLET = `
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(2048);
    this.n = 0;
    this.port.onmessage = () => {
      this.port.postMessage({ samples: this.buf.slice(0, this.n), flushed: true });
      this.n = 0;
    };
  }
  process(inputs) {
    const chans = inputs[0];
    if (!chans || !chans.length) return true;
    for (let i = 0; i < chans[0].length; i++) {
      let s = 0;
      for (const c of chans) s += c[i];
      this.buf[this.n++] = s / chans.length;
      if (this.n === this.buf.length) {
        this.port.postMessage({ samples: this.buf, flushed: false });
        this.buf = new Float32Array(2048);
        this.n = 0;
      }
    }
    return true;
  }
}
registerProcessor("wesper-capture", Capture);
`;

export interface Recording {
  samples: Float32Array;
  sampleRate: number;
}

export class Recorder {
  private stream?: MediaStream;
  private node?: AudioWorkletNode;
  private source?: MediaStreamAudioSourceNode;
  private sink?: GainNode;
  private chunks: Float32Array[] = [];
  private recording = false;
  private onFlushed?: () => void;
  deviceId = "";

  /** onLevel gets each block's RMS level in dBFS while recording. */
  constructor(private ctx: AudioContext, private onLevel: (dbfs: number) => void = () => {}) {}

  get isOpen() {
    return !!this.node;
  }

  /** Opens the microphone (asking permission the first time). Reopens if the device changed. */
  async open(deviceId = ""): Promise<void> {
    if (this.node && deviceId === this.deviceId) return;
    this.close();
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { ...CONSTRAINTS, ...(deviceId ? { deviceId: { exact: deviceId } } : {}) },
    });
    if (!Recorder.workletAdded.has(this.ctx)) {
      const url = URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" }));
      await this.ctx.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);
      Recorder.workletAdded.add(this.ctx);
    }
    this.stream = stream;
    this.deviceId = deviceId;
    this.source = this.ctx.createMediaStreamSource(stream);
    this.node = new AudioWorkletNode(this.ctx, "wesper-capture");
    this.node.port.onmessage = (e: MessageEvent<{ samples: Float32Array; flushed: boolean }>) => {
      if (this.recording || e.data.flushed) {
        this.chunks.push(e.data.samples);
        let sum = 0;
        for (const s of e.data.samples) sum += s * s;
        if (e.data.samples.length) this.onLevel(10 * Math.log10(sum / e.data.samples.length + 1e-12));
      }
      if (e.data.flushed) this.onFlushed?.();
    };
    // Connected through a muted gain to the output, so every browser keeps the worklet running.
    this.sink = new GainNode(this.ctx, { gain: 0 });
    this.source.connect(this.node).connect(this.sink).connect(this.ctx.destination);
  }
  private static workletAdded = new WeakSet<AudioContext>();

  start() {
    if (!this.node) throw new Error("microphone isn't open");
    this.chunks = [];
    this.recording = true;
  }

  /** Ends the recording, including the samples still in the worklet. */
  async stop(): Promise<Recording> {
    this.recording = false;
    if (this.node) {
      const flushed = new Promise<void>((resolve) => (this.onFlushed = resolve));
      this.node.port.postMessage("flush");
      await Promise.race([flushed, new Promise((r) => setTimeout(r, 500))]);
    }
    const n = this.chunks.reduce((k, c) => k + c.length, 0);
    const samples = new Float32Array(n);
    let off = 0;
    for (const c of this.chunks) {
      samples.set(c, off);
      off += c.length;
    }
    this.chunks = [];
    return { samples, sampleRate: this.ctx.sampleRate };
  }

  close() {
    this.recording = false;
    this.source?.disconnect();
    this.node?.disconnect();
    this.sink?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = this.node = this.source = this.sink = undefined;
  }
}

/** Audio inputs. Labels are empty until the user has allowed microphone access once. */
export async function microphones(): Promise<MediaDeviceInfo[]> {
  return (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "audioinput");
}
