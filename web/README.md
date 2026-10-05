# WESPER in the browser

A client-side version of the push-to-talk demo (`client_direct.py`). Hold the button (or Space)
while you whisper, then release: the recording is converted to normal speech and played back.
Everything runs in the page with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/),
on WebGPU when the browser has it and WebAssembly otherwise. No audio leaves the device.

- **Encoders:** Swedish (fine-tuned, the default) and the original.
- **Voices (decoders):** chosen by **Language**, then **Output model**, the vocoder: HiFi-GAN
  16 kHz or BigVGAN 22.05 kHz. Swedish is the audiobook narrator (trained by `decoder/train.py`,
  the default), for HiFi-GAN and, if that run exists, for BigVGAN. English is WESPER's Google TTS
  voice, with HiFi-GAN or through BigVGAN. Its decoder was trained for HiFi-GAN only, so the
  BigVGAN version converts its spectrograms to BigVGAN's on the way (as
  `decoder/bigvgan_preview.py` does): clearer, but a little blurred next to a voice trained for
  BigVGAN. Each voice plays and downloads at its vocoder's sample rate.
- **Compare with the other options:** each take is converted with the selected encoder and voice,
  and plays. With this on, it also lists the other encoder and the other voices, one change at a
  time, each with a **Generate** button: nothing more is converted or downloaded until you ask
  (BigVGAN voices are 600 MB). Generated rows play right away, and with keys `1`–`9`.
- **Stream:** with this on, a result starts playing when its first half second is converted, and
  the rest converts while it plays. It sounds exactly like converting it whole. With a backend
  slower than real time (BigVGAN on WASM), playback waits just long enough not to run out.
- **Backend:** Auto (WebGPU if available), WebGPU, or WASM.
- **Run again** converts an earlier take with the current settings, e.g. on the other backend.
- Each result shows its timing and the gain applied to the input. Results can be downloaded as WAV.

## Running it

```sh
# 1. Export the models to web/public/models/ (~2 min; once, and again after retraining)
.venv/bin/pip install -r web/requirements-export.txt
.venv/bin/python web/export_models.py

# 2. Start the app
cd web
bun install
bun run dev        # http://localhost:5173
```

The export takes the Swedish encoder from `colab/data/runs/n2w-finetune/encoder_best.pt`, the
Swedish narrator voice from the `decoder/runs/sv-narrator/` run folder (trained for HiFi-GAN), and
its BigVGAN version from `decoder/runs/sv-narrator-bigvgan22k/` (trained with `--vocoder
bigvgan22k`, see HOW_TO_TRAIN_DECODER.md). It skips any of them that's missing (`--sv PATH`,
`--sv-decoder FOLDER`, `--sv-bigvgan-decoder FOLDER` to use others), and fails if a run was trained
for the other vocoder. The original encoder, the English googletts decoder, HiFi-GAN and BigVGAN
are release files (downloaded on first use; cached by torch).

The models are fp32: 378 MB per encoder, about 145 MB per voice (FastSpeech2), and its vocoder:
55 MB for HiFi-GAN, which the HiFi-GAN voices share, and 450 MB for BigVGAN (NVIDIA's for English,
the Swedish narrator's own fine-tuned one), so 2.3 GB in all. Only the models the current settings
use are downloaded: 578 MB for the default Swedish encoder and narrator, and the others when a
voice is chosen or a comparison row is generated.
They're cached in the browser (Cache API) after the first download; the footer shows how much is
stored, with a link to clear it. Private windows don't allow that much storage, so there they
download on every visit, and the app says so. Files from earlier exports are deleted from the
cache on load.

## Publishing on GitHub Pages

`.github/workflows/pages.yml` builds the app on each push to the `swedish` branch that touches
`web/`, and publishes it to https://khromov.github.io/wesper-demo/. It can also be started by hand
(Actions → *Web demo on GitHub Pages* → *Run workflow*). The models are too big for the repo and
for Pages, so they live elsewhere:

1. Upload all the files in `web/public/models/` to one public folder (e.g. on S3): `models.json`,
   `encoder-*.onnx` (2), `fs2-*.onnx` (one per voice) and `vocoder-*.onnx` (one per vocoder).
   They must be publicly readable. Version 2's `decoder-*.onnx` files (FastSpeech2 and vocoder in
   one) aren't used anymore.
   After every re-export, upload `models.json` and every file whose hash in it changed.
2. Allow the site to read them (CORS). On S3:
   ```json
   [{ "AllowedOrigins": ["https://khromov.github.io", "http://localhost:5173"],
      "AllowedMethods": ["GET", "HEAD"], "AllowedHeaders": ["*"], "MaxAgeSeconds": 86400 }]
   ```
3. Point the build at the folder: `gh variable set WESPER_MODELS_URL --body https://…/folder/`.
   The workflow fails with a message until this is set.

The published site uses the Space's origin endpoint (`https://sta-public.fra1.digitaloceanspaces.com/wesper-models/`),
not its CDN, so a re-upload shows up at once. Browsers still cache the `.onnx` files: their URLs
carry their hash. Behind a CDN instead (the DigitalOcean Spaces CDN is Cloudflare), purge the
folder's cache after changing the CORS rules or re-uploading. The CDN keeps files for 7 days and
ignores `Vary: Origin`, so it can keep serving copies without the CORS header, or `models.json`
from the previous export.

To try the hosted models locally: `VITE_MODELS_URL=https://…/folder/ bun run dev`.

GitHub Pages can't send the COOP/COEP headers that WASM needs for threads, so the page loads
[coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker), which adds them. On a first
visit the page reloads once for that. Private windows get no service worker, so WASM runs
single-threaded there (WebGPU is unaffected).

## Tests

```sh
bun test src                  # level normalization, WAV, manifest, backend choice (~0.2 s)
bun run check                 # svelte-check / TypeScript
bun run test:e2e              # the app in Chrome with a fake microphone (~1-2 min, needs the models)
bun run test:e2e --pages      # the same on a build set up like GitHub Pages, models on another origin
bun run test:e2e --models DIR # with the models exported to DIR (export_models.py --out DIR)
cd .. && .venv/bin/python -m unittest tests.test_export_models tests.test_web_level_fixture -v   # ~40 s
WESPER_BIGVGAN=1 .venv/bin/python -m unittest tests.test_export_models -v                        # + BigVGAN voices, ~2 min
WESPER_EXPORT_SMOKE=1 .venv/bin/python -m unittest tests.test_export_models -v                   # + full export, ~2 min
```

Without a BigVGAN-trained run, the BigVGAN tests use a stand-in: the HiFi-GAN narrator's weights
set up for BigVGAN (`stand_in_bigvgan_run` in `tests/test_export_models.py`). It runs the same
graph, so it tests the export and the app, but it sounds rougher than a trained voice. To try the
app with it:

```sh
.venv/bin/python -c "from tests.test_export_models import stand_in_bigvgan_run as s; s('/tmp/standin')"
.venv/bin/python web/export_models.py --out /tmp/wesper-models --sv-bigvgan-decoder /tmp/standin
cd web && bun run test:e2e --models /tmp/wesper-models    # or: VITE_MODELS_URL=… with any static server
```

Never export the stand-in into `web/public/models/`.

The end-to-end test plays `sample_whisper.wav` as the microphone. It checks push-to-talk with the
mouse and with Space, file upload, both encoders, every voice (at its own sample rate), WebGPU and WASM (which must agree), that settings
survive a reload, the cache cleanup, and the private-window message. With `--pages`, the build is
served under `/wesper-demo/` without COOP/COEP headers (as GitHub Pages does), and the models from a
second origin with CORS (as S3 does); it also checks that the service worker gives WASM its threads.

## How it works

| Piece | Where |
|---|---|
| ONNX export, checked against PyTorch | `export_models.py` |
| Models list (`models.json`), file URLs | `src/lib/models/` |
| Inference in a Web Worker; downloads with progress and caching; warm-up | `src/lib/engine/worker.ts`, `src/lib/models/download.ts` |
| Streaming: the vocoder's chunks, and when to start playing | `src/lib/engine/stream.ts`, `src/lib/audio/playback.ts` |
| WebGPU or WASM | `src/lib/engine/backend.ts` |
| Microphone (AudioWorklet), resampling to 16 kHz, playback | `src/lib/audio/` |
| Input level normalization, a port of `speech_dbfs()` | `src/lib/audio/level.ts` |
| State and actions | `src/lib/app.svelte.ts` |

Conversion is the same as in `whisper_normal.py`: encoder (audio → 256-dim units every 20 ms),
then FastSpeech2 (units → mel spectrogram), then the vocoder (mel spectrogram → 16 kHz audio with
HiFi-GAN, 22.05 kHz with BigVGAN). Each is its own file, and `models.json` (version 3) gives each
voice its vocoder, sample rate and hop.

Streaming runs the encoder and FastSpeech2 on the whole take: they're cheap (FastSpeech2 is 1–7% of
the decoding time), and their transformers use context from across the utterance, so chunks of
them would sound different. The vocoder, which is most of the work, runs on chunks of the mel:
first 0.5 s, then 2 s at a time, each with 0.4 s of mel on both sides. The vocoders are
convolutional and see less than that, so the chunks join into exactly the audio of one run (the
export checks each vocoder: 130–330 dB SNR). The chunks play back to back in an AudioContext at the
voice's own sample rate, so nothing is resampled at the joins. Playback starts 20 ms after the first
chunk, or later if the first chunk's speed says the rest wouldn't keep up. For the English voice through BigVGAN, the graph also moves the decoder's
spectrogram to BigVGAN's frames and corrects each band's level in between, with the fit in
`decoder/mel_map_hifigan16k_to_bigvgan22k.json`. An encoder trained on level-normalized audio (the Swedish one) gets its input
normalized to the level it was trained on; the original gets it unchanged, as in the Python demo.

### Things that matter

- **Microphone processing is off.** Echo cancellation, noise suppression and automatic gain are
  disabled: noise suppression removes whispers, and gain changes what the encoder hears.
- **Cross-origin isolation.** WASM needs the `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp` headers to use threads. The dev and preview servers
  send them; on GitHub Pages, coi-serviceworker adds them (see above). Without them WASM runs
  single-threaded, and the app says so.
- **ONNX Runtime's JSEP build** (`ort.min.mjs` plus `public/ort/*.jsep.*`, copied on
  `bun install` by `scripts/copy-vendor.ts`). Its newer native WebGPU build gave NaN audio from the decoder in Chrome 154.
- **Models elsewhere:** `VITE_MODELS_URL` loads them from another server, which must send CORS
  headers; the build then leaves out `public/models/`. Without it, `bun run build` copies the
  models into `dist/` (1.16 GB, 1.76 GB with the BigVGAN voice).

### Measured (Chrome 154, Apple M3, 5.8 s whisper)

| Backend | Conversion | vs. PyTorch |
|---|---|---|
| WebGPU | ~1.2–1.7 s | same output (86–109 dB SNR) |
| WASM, 8 threads | ~4 s | same output |

The BigVGAN voice is much slower: about real time on WebGPU (1.4–2.0 s for a 1.9 s whisper), and
about 9× slower than real time on WASM (16–17 s for the same whisper), so it's only practical on
WebGPU. The export matches PyTorch at 72–88 dB SNR, and WebGPU matches WASM at 88 dB.

The first conversion after loading compiles GPU shaders (a few seconds); the app does that with a
second of silence while loading.

Smaller models were tried and dropped. **fp16** halved the download and the time, and looked close
on paper (units cosine 0.9998, under 1 dB log-mel difference measured on the CPU), but sounded
clearly worse in the browser. **int8** was no faster in WASM, and further from PyTorch still.

### Export notes

`export_models.py` rewrites a few operations so they export, without changing what they compute
(the tests compare each with the original):

- `nn.TransformerEncoderLayer` attention is written out. Traced, `nn.MultiheadAttention` fixes
  the sequence length in a Reshape, so only one input length works.
- `torch.bucketize` becomes a compare-and-count. FastSpeech2's length regulator (a Python loop)
  becomes a cumsum lookup. Its durations are 1 frame per unit on every clip tried, but the
  predictor still runs, as in the Python demo.
- The decoder's position table covers 120 s, the longest input the app accepts.
- For BigVGAN (11.6 ms frames), the units are interpolated to its frames inside the graph, as
  `vocoders.units_to_frames` does, so the frame count follows the input's length. Positions are
  exact integer fractions; float32 would drift by 1e-4 after a minute.
- BigVGAN's anti-aliasing filters are expanded to their channel count at run time, which the
  exporter can't size; each gets a fixed, pre-expanded kernel instead (`fixed_bigvgan_filters`).
