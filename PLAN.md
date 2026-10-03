# Plan: Swedish WESPER

How to make whisper-to-normal conversion work well in Swedish. Data details live in
[NOTES.md](NOTES.md); this file covers what to build, in which order, and on what hardware.

Nothing below is implemented in this repo yet. A decoder-training prototype was built and tested
on 2026-10-02 and then removed; its findings are kept here (marked **verified**) so they don't
have to be rediscovered.

## How the pipeline works

| Stage | Model file | What it does | Trained on |
|---|---|---|---|
| Encoder (STU, HuBERT) | `model-layer12-450000.pt` (1.1 GB) | audio → 256-dim "soft units", one per 20 ms | LibriSpeech 960 h + LPC pseudo-whispers of it + wTIMIT (English) |
| Decoder (UTS, FastSpeech2) | `googletts_neutral_best.tar` / `lambda_best.tar` | units → mel spectrogram; decides the output voice | one English voice each, no transcripts |
| Vocoder (HiFi-GAN 16 kHz) | `g_00205000` | mel → waveform | language-independent; no retraining needed |

The model files are downloaded on first run to `~/.cache/torch/hub/checkpoints/`.

From the paper: the encoder was trained HuBERT-style (masked prediction of 100 k-means units) on
a mix of normal and whispered clips that were *not* paired. That took 48 h on two RTX 6000s.
Each decoder took 26 h from scratch.

Swedish suffers at two points. The encoder only knows English sound categories, and it has only
seen English whispers. The decoder can only produce sounds it heard in English.

## Step 0: Find out where Swedish fails

Measure this before training anything; it decides which phase matters most and gives the
baseline to beat.

1. Run three kinds of input through the current pipeline (`convert.py`):
   - normal Swedish speech
   - pseudo-whispered Swedish (toWhisper and Normal2Whisper)
   - real whispered Swedish (our own recordings of known sentences)
2. Transcribe the outputs with a Swedish ASR model, e.g. KBLab's KB-Whisper. Compute the word
   error rate against the known sentences; Common Voice's `validated.tsv` has a `sentence`
   column. Also listen to a sample of the outputs.
3. Read the result:
   - **Normal input is fine, whispers are bad:** the encoder's whisper robustness is the
     problem. Phase 2 matters most.
   - **Normal input is also bad:** the problem is the decoder (accent) and/or how the encoder
     represents Swedish sounds. Start with Phase 1.

## Phase 1: Swedish decoder (fine-tune FastSpeech2)

**Data.** The decoder learns *one* voice, so it needs one speaker: normal speech, clean audio,
the same mic and room throughout, and no transcripts. Common Voice doesn't fit, because it has
thousands of speakers and mixed mics. Aim for at least 1 hour, preferably 3–10 hours; that is
a rule of thumb, not measured for this model. Candidates:
- **NST Swedish speech-synthesis corpus:** one male speaker, ~5,300 studio sentences, CC0. The
  old Språkbanken download link is dead; find it in their catalogue or the Hugging Face copy
  (`jimregan/nst_swedish_tts`).
- **Our own voice,** recorded with the Scarlett 2i2. Converted whispers then sound like us.
- **LibriVox Swedish audiobooks:** public domain, but few titles and mixed quality.

**Preprocessing (`preprocess.py`).** For each recording:
- Split at pauses into segments of at most 10 s. Training positions are capped at 1000 frames
  (20 s).
- Normalize the level (see Phase 2, *Loudness*).
- Save the following per segment, frame-aligned:
  - **units:** `whisper_normal.wav2units()`
  - **log-mel:** HiFi-GAN style
  - **pitch:** pyworld `dio` + `stonemask` at a 20 ms frame period, with unvoiced frames filled
    by interpolation
  - **energy:** the L2 norm of the STFT magnitude per frame
- Write `stats.json` as `[min, max, mean, std]` per feature (the format FastSpeech2 expects),
  with mean and std computed after removing outliers.
- Write the `train.txt` / `val.txt` lists.
- Skip segments that already exist, so the script can be re-run.

The repo has no mel code of its own (`hifigan/` only has the generator), so the extraction has
to be recreated. **Verified** settings:
- n_fft 1024, hop 320, window 1024, 80 mels, fmin 0, fmax 8000.
- Reflect-pad by (n_fft − hop) / 2 on each side, then STFT with `center=False`.
- Magnitude, mel filterbank, then `ln(clamp(x, 1e-5))`.
- This gives exactly `len(wav) // 320` frames, the same count as the HuBERT units, so units and
  mel line up 1:1.

**Model code fix (`libs/FastSpeech2/model/modules.py`).** When duration targets are given (as in
training), `VarianceAdaptor` skips the duration predictor and returns `None`. That has two
consequences:
- `FastSpeech2Loss` crashes on the `None`.
- The predictor never trains, even though inference still uses it. If the rest of the network
  shifts during fine-tuning, it can start dropping or repeating frames.

Fix it by computing `log_duration_prediction = self.duration_predictor(x, src_mask)` in the
`duration_target is not None` branch, on the same input it sees at inference. The inference
path stays unchanged.

**Training (`train.py`).**
- **Batches:** FastSpeech2's 12-tuple layout, with units in place of phoneme ids, every duration
  set to 1, and speaker 0. Group batches by length to keep padding low.
- **Starting point:** initialize from `googletts_neutral_best.tar` rather than from scratch.
  Its pitch and energy bins are stored in the checkpoint, so they override `stats.json` and
  keep the embeddings meaningful.
- **Training loop:** reuse `ScheduledOptim` and `FastSpeech2Loss` from `libs/`.
- **Checkpoints:** save `{"model", "optimizer"}` (~420 MB each). That is the format
  `whisper_normal.load_fastspeech2()` already loads, so the clients use them as-is.
- **Progress samples:** every N steps, convert a few whispered test clips and save the audio.
- **Learning rate:** the original schedule peaks at ~1e-3 after 4,000 warm-up steps. That may
  be high for fine-tuning; lower it if early samples get worse.

**Verified in the prototype:**
- **Vocoder copy synthesis:** a mel error of 0.28 with the settings above, against 0.65–2.7
  for deliberately wrong variants (±6 dB, log10).
- **Pretrained decoder on Swedish:** mel L1 of 1.84, against 1.31 on English. The error is
  mostly spectral shape (per-band offsets of −3.4…+1.5), not overall level.
- **Smoke test:** on 50 s of the macOS Alva voice, the loss fell from 8.8 to 5.8 in 60 steps.
- **Durations:** stayed at 1 frame per unit.
- **Checkpoints:** resuming worked, and `convert.py` loaded the trained checkpoint.

**Use:** `python client_direct.py --fastspeech2 path/to/checkpoint.pth.tar`.

## Phase 2: Swedish encoder (fine-tune the STU)

**Data** (see NOTES.md):
- Common Voice sv 27.0, `validated.tsv` only (~56 h, many speakers; that's fine here).
- Its toWhisper and Normal2Whisper copies.
- The 16 kHz normal versions, regenerated with the ffmpeg command in NOTES.md.
- Some real whispers, because neither tool produces real whispering: our own Swedish
  recordings, and possibly wTIMIT.
- Hold out speakers (`client_id`), not just clips, so evaluation uses unseen voices.

**Code to start from.** This repo's HuBERT code comes from `bshall/hubert`. Its training script
is the loop to adapt: masked cross-entropy on `HubertSoft` logits, which are the cosine
similarity to 100 label embeddings divided by 0.1. `units()` returns the 256-dim projection
that the decoder consumes. Initialize from `model-layer12-450000.pt`.

**Choosing the training targets.** We have *paired* data (each whisper has its normal source),
which the paper didn't use. Two options:

- **A. Paired soft-unit distillation (try first).**
  - **Setup:** a frozen copy of the current encoder (the teacher) encodes the normal clip. The
    encoder being trained (the student) learns to produce the same soft units from the
    whispered copy, with an L1 or cosine loss. It also sees normal clips, so normal-speech
    units don't drift.
  - **Pros:** the unit space stays the same, so the existing decoders and a Phase 1 decoder keep
    working without retraining. It's cheap: teacher units can be extracted once up front.
  - **Cons:** it only teaches whisper robustness. Swedish sounds the teacher already merges in
    normal speech stay merged.
- **B. Paper-style masked prediction with Swedish labels.**
  - **Setup:** run k-means (100 clusters, matching the shape the inference code loads with
    `strict=True`) on intermediate-layer features of normal Swedish. Each whispered clip gets
    its paired normal clip's labels.
  - **Pros:** the labels fit Swedish sounds.
  - **Cons:** the label embeddings must be re-learned, the unit space changes so the decoder
    must be retrained (Phase 3), and it needs more compute.

Pick B if Step 0 or the evaluation shows Swedish sounds merging even in normal speech.

**Pairing details.**
- Use the same random crop (e.g. 2–8 s) for both clips of a pair.
- Normal2Whisper outputs can be up to 10 ms longer; trim them to the normal clip's length before
  encoding.
- Mix normal, toWhisper, Normal2Whisper and real whispers in each batch.

**Loudness: normalize in training *and* at inference.** **Verified** on `sample_whisper.wav`
(−38 dBFS RMS): the encoder is *not* level-invariant.

| Gain | Clip RMS | Unit similarity to 0 dB |
|---|---|---|
| −10 dB | −48 dBFS | 0.98 |
| −20 dB | −58 dBFS | 0.93 |
| −30 dB | −68 dBFS | 0.77 |
| −40 dB | −78 dBFS | 0.19 |

- **Training data:** Normal2Whisper's quietest 5% (−63.5 dBFS and below) falls where units
  degrade. Normalize at load time as NOTES.md suggests, which also stops loudness from becoming
  a "whisper" cue.
- **Inference:** add the same normalization to `whisper_normal.py` before `wav2units()`. Mic
  input in `client_direct.py` is not normalized today.
- **Caveat:** this was measured on one clip; confirm on more.

**Evaluation.**
- **Held-out pairs:** cosine similarity of whisper vs normal soft units, and agreement of their
  argmax labels.
- **Normal speech:** student vs teacher units, to catch drift.
- **End to end:** the Step 0 word error rate, plus listening.

**Compute.** Much smaller than the original: ~56 h normal plus two whisper copies (~170 h), versus
roughly 2,000 h. HuBERT base has ~95M parameters; plan for one 24–48 GB GPU for hours to about a
day. That is an estimate, not measured.

## Phase 3: Retrain the decoder on the new encoder

Needed after option B, or if option A shifts the units noticeably. Re-run Phase 1's
preprocessing (the units change) and training. The code is reused as-is.

## Hardware and environment

- **This Mac:** fine for the demo, preprocessing, ASR scoring and listening. Not for training:
  the decoder ran at 0.9 s/step at batch 4 on the CPU (**verified**), so 100k steps at batch 32
  would take over a week.
- **GPU:** an NVIDIA card with CUDA; 24 GB is comfortable for the decoder, 24–48 GB for the
  encoder. Good fits are RTX 4090, A10G, L4 and A100. The pinned `torch==2.0.0` PyPI wheel is
  built for CUDA 11.7 and has no kernels for H100 or newer; avoid those or upgrade torch
  (untested).
- **Training time (decoder):** roughly 3–8 h for 100k steps on those GPUs (estimate). Samples
  will likely sound good earlier.
- **Python 3.11:** the pins (`torch==2.0.0`, `numpy==1.23.5`) don't install on 3.12+, which
  includes current Colab.
- **Install gotchas found while getting the demo running:**
  - `requirements.txt` lists PyYAML twice (6.0 and 6.0.1).
  - `inflect==6.0.4` needs `pydantic<2`.
  - `librosa==0.10.0` needs `pkg_resources`, so pin `setuptools<70`.
  - `pyaudio` needs PortAudio headers (`portaudio19-dev` on Linux) but isn't used by training.
  - Training adds `pyworld` for pitch.
- **Disk:**
  - models: 1.5 GB
  - decoder checkpoints: ~420 MB each
  - decoder features: ~250 MB per hour of audio
  - encoder data: see NOTES.md
- **Code:** `origin` is `rkmt/wesper-demo` (no push access). Use a fork, or copy the folder to
  the GPU machine.

## Files to add or change

| File | Phase | Change |
|---|---|---|
| `preprocess.py` | 1 | new: decoder features from single-speaker recordings |
| `train.py` | 1 | new: decoder fine-tuning |
| `config/my_preprocess16k_sv.yaml`, `config/my_train16k_sv.yaml` | 1 | new configs |
| `libs/FastSpeech2/model/modules.py` | 1 | train the duration predictor (one line) |
| `train_encoder.py` (+ data list builder) | 2 | new: paired encoder fine-tuning, adapted from bshall/hubert |
| `whisper_normal.py` | 2 | level normalization before `wav2units()` |
| `requirements-train.txt` | 1–2 | training dependencies, without `pyaudio` |
| `README.md` | 1–2 | how to train |

## Open questions

- Is the Swedish problem mostly the encoder or the decoder? Step 0 answers this.
- Which single-speaker voice to use for the decoder, and under what licence?
- How much real Swedish whispering can we record, at least for evaluation? No public Swedish
  whisper corpus was found.
- Swedish word accents (*anden* "the duck" vs *anden* "the spirit") are carried by pitch, which
  whispers lack. The decoder guesses pitch from units, so it may not recover them; evaluate this
  specifically.
