# How to train a Swedish voice for WESPER (the decoder)

WESPER has three parts:

1. The **encoder** turns speech, whispered or not, into speech units. [HOW_TO_TRAIN.md](HOW_TO_TRAIN.md)
   fine-tunes it for Swedish whispers.
2. The **decoder** turns units into a mel spectrogram, in one particular voice.
3. The **vocoder** turns the mel spectrogram into audio.

This guide trains the decoder on one speaker, so that WESPER speaks with a Swedish voice instead
of its English one. The decoder only needs recordings of that speaker, no transcripts.

You end up with a folder `decoder/runs/sv-narrator/`. Copy it into this repo and
`./client_direct_sv.sh` uses it automatically.

**Data:** `swe-audiobook/book1/`, 187 chapters (MP3), 15.1 hours, one female narrator. The
trained decoder will sound like her. That's fine for your own research, but check the rights
before you share or publish a model trained on someone's voice.

## What you need on the training computer

- **This repo,** including `libs/`, `config/`, `hifigan/` and the `decoder/` folder.
- **The data:** either the audiobook folder, or the already-prepared `decoder/data/sv-narrator/`
  if you copy that over (see step 2).
- **Python with WESPER's requirements, plus `pyworld`:**
  ```sh
  pip install -r requirements.txt pyworld "setuptools<81"
  ```
  Install the PyTorch build that matches your GPU (CUDA or ROCm). `setuptools<81` is needed
  because pyworld still imports `pkg_resources`.
- **`ffmpeg`** on the PATH.
- **A GPU** is strongly recommended: on a laptop CPU, training runs at under one step per second.
  Both scripts accept `--device cuda` (which also covers AMD GPUs with ROCm), `mps` or `cpu`.

WESPER's checkpoints (encoder, Google TTS decoder, vocoder) download automatically on first use,
about 1.5 GB in total.

## 1. Check the setup (optional, about a minute)

```sh
WESPER_DECODER_SMOKE=1 python -m unittest discover -s decoder/tests -v
```

This makes a tiny fake dataset, prepares it, trains a few steps, resumes, and converts a whisper
with the result through WESPER. If it passes, everything below will run.

## 2. Prepare the data

Skip this if you copied a finished `decoder/data/sv-narrator/` (it has a `stats.json`).

```sh
python decoder/prepare_data.py swe-audiobook/book1 decoder/data/sv-narrator --device cuda
```

For each chapter it:
1. Cuts the chapter into utterances of 1–12 s at pauses.
2. Normalizes each utterance to −20 dBFS speech level.
3. Computes, for every 20 ms: the speech units (the decoder's input), the mel spectrogram (its
   target), the pitch and the energy.

The last 3 chapters become the validation set. If it's interrupted, run it again: finished
chapters are skipped. On a laptop CPU it takes about an hour; it's faster with a GPU.

## 3. Train

```sh
python decoder/train.py decoder/data/sv-narrator decoder/runs/sv-narrator --device cuda
```

It starts from WESPER's Google TTS decoder and adapts it to the new voice, which needs far fewer
steps than starting from scratch. Every 100 steps it prints a line like this (the numbers here
are made up):

```
step 2400/100000  loss 1.203 (mel 0.381, postnet 0.380, pitch 0.312, energy 0.088)  lr 7.7e-04  4.10 steps/s  ETA 397 min
```

Every 2,000 steps it measures the validation loss and saves `decoder_best.pt` when the loss
improves (`-> saved decoder_best.pt`). It also synthesizes validation utterances you can listen
to, in `decoder/runs/sv-narrator/samples/`:

- `reference/`: the narrator's actual recording.
- `vocoded-target/`: her recording's own mel spectrogram through the vocoder. This is the best a
  decoder could possibly sound with this vocoder.
- `step_NNNNNN/`: the decoder at that step, from speech units only, exactly as WESPER uses it.

**Stopping and resuming:** stop whenever the newest samples sound good; `decoder_best.pt` is
already saved. If training stops for any reason, run the same command again and it resumes from
the last save, every 2,000 steps by default.

| Option | Default | When to change it |
|---|---|---|
| `--steps` | 100000 | Train longer or shorter. |
| `--batch-size` | 16 | Lower it (e.g. to 8) if the GPU runs out of memory. |
| `--init` | `googletts` | Start from `lj` (WESPER's LJSpeech voice), `none` (from scratch, which needs far more steps), or a checkpoint path. |
| `--eval-every`, `--save-every` | 2000 | How often to validate and save samples, and how often to save a resumable checkpoint. |
| `--samples` | 6 | How many validation utterances to synthesize at each validation. |
| `--workers` | 2 | Data-loading processes. |

## 4. Use the voice

Copy these three files from the training computer into this repo's `decoder/runs/sv-narrator/`:
`decoder_best.pt`, `preprocess.yaml` and `stats.json`.

- **GUI:** run `./client_direct_sv.sh`. It picks up the decoder automatically; the log shows
  `decoder: decoder_best.pt`.
- **Anything else in WESPER:** pass
  `--fastspeech2 decoder/runs/sv-narrator/decoder_best.pt --preprocess_config decoder/runs/sv-narrator/preprocess.yaml`.

## How it works, compared with WESPER's own training

WESPER's decoder training code is public, in Jun Rekimoto's
[rkmt/UnitFastSpeech2](https://github.com/rkmt/UnitFastSpeech2). These scripts were checked
against it at the WESPER-era commit `bd3c317` (August 2022).

**The same as WESPER:**
- **Units:** the decoder's input is WESPER's original encoder (`model-layer12-450000.pt`, frozen)
  applied to the speaker's normal speech. The fine-tuned Swedish encoder was trained to produce
  these same units from whispers, so the decoder gets the input it was trained on.
- **Durations:** one unit per mel frame, with the duration predictor trained toward that. WESPER
  relies on the duration predictor at inference. The copy of FastSpeech2 in this repo skipped it
  during training, so the loss couldn't run; one line in `libs/FastSpeech2/model/modules.py` now
  runs it on its inference input. The inference path is unchanged.
- **Pitch and energy:**
  - Pitch uses WORLD's DIO + StoneMask, with unvoiced stretches interpolated, and utterances
    with no voiced frames are dropped. Energy is the norm of each frame's spectrum.
  - Both are normalized over the whole dataset: mean and std fitted on each utterance's values
    without outliers, and `stats.json` in FastSpeech2's [min, max, mean, std] format.
  - When starting from the Google TTS decoder, its pitch and energy bins are replaced by the new
    speaker's.
  - The narrator's average pitch is about 137 Hz. Three different pitch trackers agree, so that's
    her voice, not a tracking error.
- **Model, loss and learning-rate schedule:** the repo's own FastSpeech2 code and
  `config/my_train16k_LJ.yaml`: Adam, warmup 4,000 steps, peak learning rate about 1e-3,
  gradient clipping at 1.0.

**Deliberately different:**

| | WESPER | Here | Why |
|---|---|---|---|
| Mel framing | FastSpeech2's `TacotronSTFT` (centered on sample t·320) | HiFi-GAN's (centered on t·320 + 160) | WESPER's own vocoder expects HiFi-GAN framing: round-trip error 0.24 vs 0.31–0.35, and Tacotron framing comes out about 130–160 samples late. HiFi-GAN framing also lines up exactly with the units. Scale and log are the same. |
| Unit padding | padded twice in training (shifted 40 samples) | padded once | Matches how WESPER computes units at inference. |
| Starting point | from scratch per voice (the released decoders: 114k–421k steps, batch 32) | fine-tuned from the Google TTS decoder | Much faster. Use `--init none` (and more steps) to train from scratch like WESPER. |
| Loudness | none: audio used as loaded | each utterance at −20 dBFS speech level | The same level WESPER's input is normalized to here. The paper doesn't mention levels. |
| Best checkpoint | lowest loss on a single training batch | lowest validation loss | Validation is a more reliable measure. |

## Tests

```sh
python -m unittest discover -s decoder/tests -v                           # ~3 s
WESPER_DECODER_SMOKE=1 python -m unittest discover -s decoder/tests -v    # + real runs, ~1 min
```

The tests cover:
- the mel computation against HiFi-GAN's
- frames lining up with units
- pitch tracking and gap filling
- cutting only in pauses, with long speech split and no overlaps
- the stats format and the batch layout
- the duration-predictor fix
- an end-to-end run of both scripts, including resume, re-run skipping, an untouched source
  folder, and converting audio through WESPER with the new decoder
