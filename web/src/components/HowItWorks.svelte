<script lang="ts">
  // How a whisper becomes speech: the three models, with the ones selected above. Starts collapsed.
  import { vocoderLabel, type DecoderEntry, type EncoderEntry } from "../lib/models/manifest";

  let { encoder, decoder }: { encoder: EncoderEntry | null; decoder: DecoderEntry | null } = $props();

  const khz = (rate: number) => `${+(rate / 1000).toFixed(2)} kHz`;
</script>

<details class="card how" data-testid="how-it-works">
  <summary>How it works</summary>

  <ol class="flow" aria-label="The conversion, step by step">
    <li class="box end">
      <span class="step">In</span>
      <span class="name">Your whisper</span>
      <span class="sub">16 kHz{encoder?.targetDbfs != null ? `, leveled to ${encoder.targetDbfs} dBFS` : ""}</span>
    </li>
    <li class="arrow" aria-hidden="true"></li>
    <li class="box model">
      <span class="step">1 · Encoder</span>
      <span class="name" data-testid="how-encoder">{encoder?.label ?? "…"}</span>
      <span class="sub">HuBERT-soft → speech units</span>
    </li>
    <li class="arrow" aria-hidden="true"></li>
    <li class="box model">
      <span class="step">2 · Decoder</span>
      <span class="name" data-testid="how-decoder">{decoder?.label ?? "…"}</span>
      <span class="sub">FastSpeech2 → mel spectrogram{decoder?.melMap ? ", converted for BigVGAN" : ""}</span>
    </li>
    <li class="arrow" aria-hidden="true"></li>
    <li class="box model">
      <span class="step">3 · Vocoder</span>
      <span class="name" data-testid="how-vocoder">{decoder ? vocoderLabel(decoder.vocoder) : "…"}</span>
      <span class="sub">{decoder?.vocoderFineTuned ? "fine-tuned for this voice" : decoder?.vocoder === "bigvgan22k" ? "NVIDIA's" : "WESPER's"} → audio</span>
    </li>
    <li class="arrow" aria-hidden="true"></li>
    <li class="box end">
      <span class="step">Out</span>
      <span class="name">Normal speech</span>
      <span class="sub">{decoder ? khz(decoder.sampleRate) : ""}</span>
    </li>
  </ol>

  <dl class="small">
    <dt>1 · Encoder</dt>
    <dd>
      Turns the audio into <em>speech units</em>, one every 20 ms: what is said, but not who says it. The Swedish
      encoder is WESPER's, fine-tuned so that a Swedish whisper gives the same units as normal speech.
    </dd>
    <dt>2 · Decoder</dt>
    <dd>
      Turns the units into a mel spectrogram in one speaker's voice, predicting its pitch and loudness. This is
      what <em>Language</em> picks: the Swedish narrator was trained on 14 hours of one audiobook, the English
      voice is WESPER's Google TTS one.
    </dd>
    <dt>3 · Vocoder</dt>
    <dd>
      Turns the mel spectrogram into audio. This is what <em>Output model</em> picks: WESPER's HiFi-GAN at 16 kHz, or
      NVIDIA's BigVGAN at 22.05 kHz, which is clearer but a larger download and slower.
      {#if decoder?.vocoderFineTuned}
        This voice's BigVGAN is fine-tuned on its decoder's own mel spectrograms, which are smoother than real ones
        and otherwise come out with an electric buzz.
      {:else if decoder?.melMap}
        This voice's decoder was trained for HiFi-GAN, so its mel spectrogram is converted to BigVGAN's on the way,
        which blurs it a little.
      {/if}
    </dd>
  </dl>
  <p class="small muted">
    All three run here in the browser, on WebGPU or WebAssembly: your audio never leaves this device.
  </p>
</details>

<style>
  .how {
    padding: 12px 16px;
  }
  summary {
    cursor: pointer;
    font-weight: 600;
  }
  .flow {
    list-style: none;
    margin: 14px 0 6px;
    padding: 0;
    display: flex;
    align-items: stretch;
    gap: 4px;
  }
  .box {
    flex: 1 1 0;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 1px;
    padding: 8px 10px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface-2);
  }
  .box.model {
    background: var(--accent-soft);
    border-color: transparent;
  }
  .box.end {
    flex: 0 0 auto;
    max-width: 84px;
    padding: 8px 2px;
    border: none;
    background: none;
    justify-content: center;
    text-align: center;
  }
  .step {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .name {
    font-weight: 600;
    font-size: 14px;
    line-height: 1.3;
    overflow-wrap: break-word;
  }
  .sub {
    font-size: 12px;
    color: var(--muted);
  }
  .arrow {
    flex: none;
    width: 22px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: var(--muted);
    font-size: 11px;
    line-height: 1.2;
    text-align: center;
  }
  .arrow::after {
    content: "→";
    font-size: 18px;
    line-height: 1;
  }
  dl {
    margin: 12px 0 0;
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 6px 12px;
  }
  dt {
    font-weight: 600;
    white-space: nowrap;
  }
  dd {
    margin: 0;
  }
  p {
    margin: 10px 0 0;
  }
  @media (max-width: 700px) {
    .flow {
      flex-direction: column;
    }
    .box.end {
      max-width: none;
      text-align: left;
      padding: 4px 10px;
    }
    .arrow {
      width: auto;
      padding: 2px 0;
    }
    .arrow::after {
      content: "↓";
    }
    dl {
      grid-template-columns: 1fr;
    }
  }
</style>
