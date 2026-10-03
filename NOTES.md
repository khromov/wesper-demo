# Notes: Swedish data for WESPER

Working notes for adapting WESPER to Swedish. We fine-tune WESPER's speech-to-unit encoder on
Swedish normal speech paired with **Normal2Whisper** pseudo-whispers. We chose Normal2Whisper
after listening to both methods. The existing decoder stays, so the output voice is still
WESPER's English voice. A Swedish decoder would be a later, separate step. These notes record
things that are easy to get wrong.

## Datasets

| | Path | Format | Status |
|---|---|---|---|
| Source | `../swedish-common-voice-cv-corpus-27.0-2026-09-11/clips/` | 50,279 mp3, 48 kHz mono, 56.2 h | Untouched |
| toWhisper | `../toWhisper/sv-cv27-whisper16k/clips/` | 50,279 wav, 16 kHz mono PCM16, 6.1 GB | Done 2026-10-03 |
| Normal2Whisper | `../Normal2Whisper/sv-cv27-whisper16k/clips/` | 50,279 wav, 16 kHz mono PCM16, 6.1 GB | Done 2026-10-03 |
| Training set | `colab/data/wesper-sv-n2w.tar` (+ same files in `colab/data/export/`) | 37,153 processed pairs, 16 kHz FLAC, 3.8 GB | Done 2026-10-03 |

All outputs keep the source basename (`common_voice_sv-SE_<id>`), so the corpus TSVs map onto
them directly. `colab/data/` is excluded from git locally, in `.git/info/exclude`.

## Training set

Built by `colab/prepare_data.py` from validated clips:
- **Splits:** train is every speaker outside Common Voice's test set: 36,657 clips, 39.0 h, 112
  speakers. Two speakers recorded about half of those clips; the notebook evens that out when
  sampling. Val is 496 clips, one per test-set speaker, so validation measures unseen voices.
- **Skipped:** 56 near-silent clips (speech below −45 dBFS).
- **Processing:** each clip's normal and Normal2Whisper versions are processed identically and
  stay sample-aligned:
  1. Mute the touchpad thud at the start (see below).
  2. Normalize to −20 dBFS speech level.
  3. Store 12 dB lower (−32 dBFS) so 16-bit FLAC doesn't clip. Normalized whispers peak up to
     about +13 dBFS. The notebook adds the 12 dB back when loading; `prep.json` in the tar
     records these settings. Clipped samples remain in only 13 clips, at most 42 samples each.
- **Manifest:** `manifest.tsv` lists each clip's split, speaker, original speech levels, muted
  thud length and sentence.

## The encoder is very level-sensitive: normalize everything

- **Measured on WESPER's original encoder:** turning the input down by 10 dB changes its output
  units by about 30%; by 40 dB, about 80%. Its first normalization layer has a small constant
  that dominates at speech levels, so overall gain doesn't cancel out.
- **Common Voice varies a lot:** speech level runs from −36.8 to −12.9 dBFS (1st–99th
  percentile, median −18.9).
- **Normal2Whisper is much quieter:** about 17 dB below its source, and about 11 dB below
  toWhisper.
- **What we do:** normalize every training clip, normal and whispered, to −20 dBFS speech level.
  The level estimate (`speech_dbfs()`) averages 20 ms frames after dropping silence (more than
  30 dB below the loudest frames), and caps each frame so a click can't dominate.
  - Tested on real clips: adding 2 s of silence moves it 0.0 dB, trimming to speech moves it at
    most 0.9 dB, and a loud click moves it at most 1.5 dB.
  - Normalization never amplifies more than 40 dB.
- **At inference too:** fine-tuned checkpoints record `TARGET_DBFS` and `MAX_GAIN_DB` in their
  `config`. `whisper_normal.py` (`load_hubert()`, `MyWhisper2Normal.whisper2normal()`) then
  normalizes all input to that level, using the same `speech_dbfs()`. This covers the GUI,
  `server.py` and `convert.py`. With the fine-tuned encoder, a whisper 20 dB quieter gives the
  same output, to within one 16-bit step. The original encoder has no `config`, so its input is
  unchanged.

## Touchpad thuds at the start of clips

Most Common Voice clips start with the sound of the click that started the recording: a sharp
transient right after the MP3's leading digital silence (median 37 ms), peaking about 20 dB
below speech, decaying within 50–100 ms, then quiet before the speech.

- **Rule (`thud_end()`):** if the first sound ends within 200 ms and at least 40 ms of quiet
  follows, everything before the quiet is muted, with a 5 ms fade-in. "Quiet" means near the
  background noise and at least 45 dB below the speech.
  - Speech that starts straight away has no such gap, so it's left alone.
  - A cut, rather than a mute, would break the alignment with the whispered version.
- **Effect:** muted in 47% of training clips, median 105 ms including the leading digital
  silence.
- **Known limits:**
  - A clip whose first word is shorter than 200 ms and followed by a pause would be muted too.
    The 12 plotted examples I checked by eye looked right, but this hasn't been checked
    systematically; listen to some Normal (train) clips in the player to confirm.
  - Clicks at the *end* of clips (stopping the recording) are not handled.

## Listening: corpus player

- **Where the files are:** everything is copied into the corpus repo under `clips_whispered/`
  (`towhisper/`, `normal2whisper/`, `train-normal/`, `train-n2w/`). They're APFS clones, so
  they take almost no extra disk space. The folder is excluded from git locally, in
  `.git/info/exclude`.
- **Running it:** `cd player && bun dev` (or `bun start`), then open http://localhost:3000.
  `bun dev` hot-reloads, and exits if a code edit is caught half-saved; just start it again.
- **Buttons:** five per clip: **Normal** (blue), **toWhisper** (orange), **Normal2Whisper**
  (purple), **Normal (train)** (teal) and **N2W (train)** (pink), on keys `1`–`5`.
  - Each plays its version from the start and stops anything else playing.
  - The two train versions are the exact files the model trains on. The player plays them 12 dB
    louder, which undoes the storage headroom, so you hear them at the training level.
- **Missing files:** a clip without a file for some version gets a disabled button. The train
  versions exist only for the 37,153 clips in the training set.
- **New files:** the player rescans `clips_whispered/` when the page loads (at most every 5 s).
- **Adding another version:** add one entry to `WHISPER_VARIANTS` in `player/src/shared.ts`
  (folder, label, colors, file extension, playback gain).

## Training on Colab: `colab/wesper_sv_encoder_finetune.ipynb`

- **What it does:** fine-tunes the encoder by matching the original. A frozen copy of WESPER's
  original encoder (the teacher) turns each normal clip into units. The encoder being trained
  gets the whispered version 75% of the time, otherwise the normal one, and learns to produce
  the teacher's units.
  - Why this way: the units keep the original format, so WESPER's decoder and vocoder keep
    working, and `convert.py --hubert encoder_best.pt` works as-is.
  - The checkpoint shows WESPER's encoder was trained with the soft-HuBERT training script, but
    not its exact targets, so we match the original encoder instead of guessing them.
- **To run it:** follow [HOW_TO_TRAIN.md](HOW_TO_TRAIN.md). In short: put the tar in a Drive
  folder, open the notebook in Colab with a GPU, set `DRIVE_DIR`, and run all. Checkpoints go to
  `<DRIVE_DIR>/runs/<RUN_NAME>/`, and re-running resumes after a disconnect.
- **Outputs:** the notebook ends with before/after listening on unseen speakers, and a cell to
  upload your own whisper recording.
- **Defaults:** 10,000 steps, batch 16 × 3 s crops, learning rate 3e-5 with warmup and cosine
  decay, ±3 dB random gain. Validation is the distance to the teacher's normal-speech units, for
  whispered input (should drop) and normal input (should stay near 0).

## Decoder: a Swedish voice

- **Guide and scripts:** [HOW_TO_TRAIN_DECODER.md](HOW_TO_TRAIN_DECODER.md),
  `decoder/prepare_data.py` and `decoder/train.py`.
- **Data:** `swe-audiobook/book1/`, 187 chapters, 15.1 h, one female narrator (average pitch about
  137 Hz).
- **WESPER's own training code** is [rkmt/UnitFastSpeech2](https://github.com/rkmt/UnitFastSpeech2),
  checked at commit `bd3c317`. It matches on units (the original encoder, frozen), durations
  (1, with the duration predictor trained), and pitch and energy.
- **Differs on purpose:**
  - mel framing (HiFi-GAN's, which WESPER's vocoder expects)
  - single unit padding, as at inference
  - fine-tuning from the Google TTS decoder instead of training from scratch
  - −20 dBFS level
  - best checkpoint by validation loss
- **`libs/FastSpeech2/model/modules.py`:** one-line fix. The duration predictor was skipped
  whenever durations were given, so the training loss crashed. Inference is unchanged.
- **WESPER's `vocoder_infer`** casts to int16 with `astype`, which wraps around rather than clips
  above full scale. `decoder/train.py` writes its samples from the vocoder's float output instead.
- **The English decoder is the bottleneck:** normal Swedish speech through WESPER's original
  encoder and English decoder transcribes at 37.5% CER, against 2.4% for the recordings
  themselves.

## Tests

```sh
.venv/bin/python -m unittest discover -s colab/tests -v              # packer + notebook helpers (~3 s)
WESPER_NOTEBOOK_SMOKE=1 .venv/bin/python -m unittest discover -s colab/tests -k Smoke -v
       # runs the whole notebook on CPU on fake data, incl. resume: ~40 s once WESPER's checkpoints
       # are cached. A watchdog dumps stacks and exits after WESPER_NOTEBOOK_SMOKE_TIMEOUT s (900).
(cd ../toWhisper && python3 -m unittest discover -s tests -v)                       # ~1 s
(cd ../Normal2Whisper && .venv/bin/python -m unittest discover -s tests -v)         # ~6 s
```

- **Packer tests:** the level estimator, thud detection (thud found; speech starts, long sounds
  and thuds without a gap left alone), processing (levels, alignment, muting both versions, gain
  cap, clipping), and an end-to-end run on a fake Common Voice corpus. The end-to-end run checks
  speaker-disjoint splits, skipped near-silent clips, tar = export byte-for-byte, sources
  untouched, and refusal to write inside the sources.
- **Notebook tests:** its copy of `speech_dbfs()` matches `prepare_data.py` exactly, and loading
  undoes the headroom. The smoke test runs every cell, checks that `encoder_best.pt` loads with
  WESPER's own `load_hubert()` (what `convert.py --hubert` uses), and checks that training
  resumes.
- **Converter tests:** output format, re-run skipping, failure listing, the output-inside-source
  guard, and an untouched source. For toWhisper, a regression test for the leading-silence fix:
  upstream outputs pure silence on that clip. For Normal2Whisper, a check that a voiced tone
  comes out unvoiced.

## Things to know before using the data

- **Use only clips in `validated.tsv`.** The training set already does. Six source clips are
  empty recordings (−91 dB): ids 39612325–39612329 and 30633349. toWhisper outputs exact zeros
  for them, and Normal2Whisper outputs −93 dB noise. None of them are in `validated.tsv`.
- **Lengths:** toWhisper outputs have exactly the source length. Normal2Whisper outputs are 1–6
  ms longer (median 4 ms), because WORLD works in 5 ms frames. The training set trims each pair
  to the same length.
- **Resampling:** raw 16 kHz normal-speech versions are not stored; the training set has
  processed ones. Common Voice mp3s are resampled with ffmpeg:
  ```sh
  ffmpeg -nostdin -v error -i IN.mp3 -ac 1 -ar 16000 -c:a pcm_s16le \
      -map_metadata -1 -fflags +bitexact -flags +bitexact OUT.wav
  ```
  Normal2Whisper resampled with librosa (soxr) instead. The difference is negligible, but the
  two aren't sample-identical.
- **Clipping in the whisper sets:** in toWhisper, 72 files have some clipped samples, mostly
  where the source already peaks at 0 dB. In Normal2Whisper, 5 do.
- **Neither set is real whispering.** toWhisper keeps the formants and only swaps the voice
  source for noise. Normal2Whisper also removes the glottal source (GFM-IAIF) and widens the
  formant bandwidth. Neither models the raised formants or slower pace of real whispers, so mix
  in some real whispers (wTIMIT, or our own recordings) if possible. No public Swedish whisper
  corpus was found.

## Tool setup (local changes)

- **toWhisper** (`../toWhisper`, zeta-chicken/toWhisper): `toWhisper.c` has a local fix.
  Upstream outputs a completely silent file when a clip contains 20 ms or more of exact digital
  zeros, which is common at the start of Common Voice clips. The fix skips all-zero frames. A
  fresh clone does **not** include it. The batch script is `convert_corpus.sh`, and the tests
  are in `tests/`.
- **Normal2Whisper** (`../Normal2Whisper`, chaufanglin/Normal2Whisper): uses unchanged
  `utils.pseudo_whisper_gen`. `.venv` needs `setuptools<81`, because pyworld still imports
  `pkg_resources`. The batch driver is `convert_corpus.py`, and the tests are in `tests/`.
  Upstream's `data_gen.py` doesn't fit this corpus: it makes one directory per clip and runs on
  a single core.

Both batch scripts only read the source folder, refuse an output folder inside it, write
atomically, and skip finished clips when re-run:

```sh
../toWhisper/convert_corpus.sh SRC_CLIPS_DIR OUT_DIR [JOBS]                  # ~14 min on 8 cores
../Normal2Whisper/.venv/bin/python ../Normal2Whisper/convert_corpus.py \
    SRC_CLIPS_DIR OUT_DIR --jobs 8                                           # ~2 h on 8 cores
.venv/bin/python colab/prepare_data.py CORPUS_DIR N2W_CLIPS_DIR OUT_TAR --export-dir DIR   # ~7 min
```
