# WESPER in the browser

A client-side version of the push-to-talk demo (`client_direct.py`). Hold the button (or Space)
while you whisper, then release: the recording is converted to normal speech and played back.
Everything runs in the page with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/),
on WebGPU when the browser has it and WebAssembly otherwise. No audio leaves the device.

- **Encoders:** Swedish (fine-tuned, the default) and the original, side by side. With *Also
  convert with…* on, each take is converted by both, so you can A/B them with keys `1` and `2`.
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

The export takes the Swedish encoder from `colab/data/runs/n2w-finetune/encoder_best.pt` and
skips it if that file is missing (`--sv PATH` to use another). The original encoder and the
googletts decoder are WESPER's release files (downloaded on first use; cached by torch).

The models are fp32: 378 MB per encoder and 200 MB for the decoder, so 956 MB with both encoders.
They're cached in the browser (Cache API) after the first download; the footer shows how much is
stored, with a link to clear it. Private windows don't allow that much storage, so there they
download on every visit, and the app says so. Files from earlier exports are deleted from the
cache on load.

## Publishing on GitHub Pages

`.github/workflows/pages.yml` builds the app on each push to the `swedish` branch that touches
`web/`, and publishes it to https://khromov.github.io/wesper-demo/. It can also be started by hand
(Actions → *Web demo on GitHub Pages* → *Run workflow*). The models are too big for the repo and
for Pages, so they live elsewhere:

1. Upload the four files in `web/public/models/` to one public folder (e.g. on S3):
   `models.json`, `encoder-sv.onnx`, `encoder-original.onnx` and `decoder-googletts.onnx`. After
   every re-export, upload all four again: `models.json` holds the other files' sizes and hashes.
2. Allow the site to read them (CORS). On S3:
   ```json
   [{ "AllowedOrigins": ["https://khromov.github.io", "http://localhost:5173"],
      "AllowedMethods": ["GET", "HEAD"], "AllowedHeaders": ["*"], "MaxAgeSeconds": 86400 }]
   ```
3. Point the build at the folder: `gh variable set WESPER_MODELS_URL --body https://…/folder/`.
   The workflow fails with a message until this is set.

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
cd .. && .venv/bin/python -m unittest tests.test_export_models tests.test_web_level_fixture -v   # ~40 s
WESPER_EXPORT_SMOKE=1 .venv/bin/python -m unittest tests.test_export_models -v                   # + full export, ~2 min
```

The end-to-end test plays `sample_whisper.wav` as the microphone. It checks push-to-talk with the
mouse and with Space, file upload, both encoders, WebGPU and WASM (which must agree), that settings
survive a reload, the cache cleanup, and the private-window message. With `--pages`, the build is
served under `/wesper-demo/` without COOP/COEP headers (as GitHub Pages does), and the models from a
second origin with CORS (as S3 does); it also checks that the service worker gives WASM its threads.

## How it works

| Piece | Where |
|---|---|
| ONNX export, checked against PyTorch | `export_models.py` |
| Models list (`models.json`), file URLs | `src/lib/models/` |
| Inference in a Web Worker; downloads with progress and caching; warm-up | `src/lib/engine/worker.ts`, `src/lib/models/download.ts` |
| WebGPU or WASM | `src/lib/engine/backend.ts` |
| Microphone (AudioWorklet), resampling to 16 kHz, playback | `src/lib/audio/` |
| Input level normalization, a port of `speech_dbfs()` | `src/lib/audio/level.ts` |
| State and actions | `src/lib/app.svelte.ts` |

Conversion is the same as in `whisper_normal.py`: encoder (audio → 256-dim units every 20 ms),
then FastSpeech2 and HiFi-GAN (units → 16 kHz audio). The export puts FastSpeech2 and HiFi-GAN in
one decoder file. An encoder trained on level-normalized audio (the Swedish one) gets its input
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
  models into `dist/` (956 MB).

### Measured (Chrome 154, Apple M3, 5.8 s whisper)

| Backend | Conversion | vs. PyTorch |
|---|---|---|
| WebGPU | ~1.2–1.7 s | same output (86–109 dB SNR) |
| WASM, 8 threads | ~4 s | same output |

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
