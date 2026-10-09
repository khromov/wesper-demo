# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## This is the Swedish fork

This repo (`origin`: khromov/wesper-demo, working branch `swedish`) is a fork of Jun Rekimoto's
WESPER demo (`upstream`: rkmt/wesper-demo), which is English-only. **The goal here is
whisper-to-normal conversion for Swedish.** Don't conflate the two:

- "Swedish encoder" = WESPER's HuBERT encoder fine-tuned on Swedish Common Voice + Normal2Whisper
  pseudo-whispers (`colab/`). "Original encoder" = upstream's English one.
- "Swedish voice/decoder" = FastSpeech2 trained on a Swedish audiobook narrator (`decoder/`,
  data in `swe-audiobook/book1/`). English voices (googletts, LJSpeech) are upstream's and are kept
  only as a baseline/comparison.
- New work, defaults and docs should target Swedish; the English paths must keep working
  unchanged (e.g. the original encoder gets no input-level normalization).
- `README.md`'s top half and `convert.py`/`server.py`/`client.py` are mostly upstream. Fork-specific
  docs: `HOW_TO_TRAIN.md` (encoder), `HOW_TO_TRAIN_DECODER.md` (decoder/vocoder),
  `NOTES.md` (data decisions and pitfalls), `web/README.md` (browser app).
  `PLAN.md` is the original plan; parts of it (e.g. "nothing implemented yet") are out of date.

## Pipeline (shared by Python and the web app)

audio 16 kHz → **encoder** (HuBERT-soft, 256-dim units per 20 ms) → **decoder** (FastSpeech2,
units → mel, durations fixed at 1 unit = 1 frame) → **vocoder** (HiFi-GAN 16 kHz, or BigVGAN
22.05 kHz) → audio.

- `whisper_normal.py` is the core inference module (`load_hubert`, `load_fastspeech2`,
  `units2wav`, `MyWhisper2Normal`); `convert.py`, `server.py`, `client_direct.py` all go through it.
  It imports `libs/FastSpeech2` via `sys.path`, so run scripts from the repo root.
- **Level normalization is essential.** The encoder is very loudness-sensitive. Fine-tuned
  checkpoints store `TARGET_DBFS`/`MAX_GAIN_DB` in their `config`; `whisper_normal.py` then
  normalizes input with `speech_dbfs()` from `colab/prepare_data.py`. `web/src/lib/audio/level.ts`
  is a port of the same function, checked against a Python-generated fixture
  (`tests/test_web_level_fixture.py`) — keep them in sync.
- **Vocoder selection is data-driven** (`vocoders.py`): a decoder run's `preprocess.yaml` names its
  vocoder (`vocoder: {name, checkpoint?}`); decoders without it (upstream's) use HiFi-GAN. A
  decoder only works with the vocoder its data was prepared for (`decoder/prepare_data.py --vocoder`).
  For BigVGAN, units are interpolated to its 11.6 ms frames (`vocoders.units_to_frames`).
- Decoder runs live in `decoder/runs/<name>/` (`decoder_best.pt`, `preprocess.yaml`, `stats.json`);
  the GUI's Voice dropdown and `web/export_models.py` discover them there. The Swedish encoder is
  expected at `colab/data/runs/n2w-finetune/encoder_best.pt`. Upstream checkpoints download to
  torch's hub cache on first use.
- `colab/data/`, `decoder/data/`, `decoder/runs/` are excluded via `.git/info/exclude` (large,
  local-only); `swe-audiobook` is gitignored.
- `libs/` holds vendored, modified third-party code (FastSpeech2, HuBERT, BigVGAN).

## Python

Use the repo venv (`.venv/bin/python`, Python 3.11, uv-managed). Device auto-selects CUDA or CPU.

```sh
./client_direct_sv.sh                      # GUI: Swedish encoder + decoder/runs/sv-narrator
DECODER=decoder/runs/sv-narrator-bigvgan22k-vft ./client_direct_sv.sh   # another voice run
.venv/bin/python convert.py --hubert colab/data/runs/n2w-finetune/encoder_best.pt --input in.wav --output out.wav
```

Tests are `unittest` (no pytest), in three suites:

```sh
.venv/bin/python -m unittest discover -s tests -v          # normalization, GUI, vocoders, export
.venv/bin/python -m unittest discover -s colab/tests -v    # data packer, notebook
.venv/bin/python -m unittest discover -s decoder/tests -v  # decoder/vocoder training
.venv/bin/python -m unittest tests.test_vocoders.<Class>.<test_method> -v   # single test
```

Slow/opt-in tests are gated by env vars: `WESPER_NOTEBOOK_SMOKE=1` (colab),
`WESPER_DECODER_SMOKE=1` (decoder), `WESPER_BIGVGAN=1` and `WESPER_EXPORT_SMOKE=1` (`tests.test_export_models`).

Training: the encoder trains on Colab (`colab/wesper_sv_encoder_finetune.ipynb`); the decoder and
BigVGAN fine-tune via `decoder/train.py` / `decoder/finetune_vocoder.py` — see the HOW_TO docs.
On the ROCm training server, GPU scripts must be run through `decoder/rocm.py` (MIOpen hangs).

## Web app (`web/`, Svelte 5 + Vite + Bun, ONNX Runtime Web)

Browser port of the push-to-talk demo; inference in a Web Worker on WebGPU or WASM.

```sh
.venv/bin/python web/export_models.py      # ONNX models → web/public/models/ (+ models.json)
cd web && bun install && bun run dev       # http://localhost:5173
bun test src                               # unit tests; single file: bun test src/lib/audio/level.test.ts
bun run check                              # svelte-check / TypeScript
bun run test:e2e                           # Chrome + fake mic, needs exported models
```

- Encoder, FastSpeech2 and vocoder are separate ONNX files; `models.json` (version 3) maps each
  voice to its vocoder, sample rate and hop. `export_models.py` rewrites some ops so they export
  and checks every model against PyTorch — see "Export notes" in `web/README.md` before changing it.
- Only the vocoder is streamed in chunks; encoder and FastSpeech2 run on the whole take.
- Uses ORT's **JSEP** build (copied by `scripts/copy-vendor.ts` on install), not the native
  WebGPU build (NaN audio). Mic processing (AGC/noise suppression) must stay off. fp16/int8 models
  were tried and rejected.
- Never export the test stand-in BigVGAN run into `web/public/models/`.
- `.github/workflows/pages.yml` deploys to GitHub Pages on push to `swedish` touching `web/`;
  models are hosted separately (`WESPER_MODELS_URL` / `VITE_MODELS_URL`).
