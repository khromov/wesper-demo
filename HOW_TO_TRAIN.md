# How to train the Swedish WESPER encoder

This guide walks through fine-tuning WESPER's encoder on Google Colab, so that Swedish whispers
are converted as well as possible. You end up with `encoder_best.pt`, a drop-in replacement for
WESPER's encoder: `python convert.py --hubert encoder_best.pt ...`. The output voice stays
WESPER's existing English voice. For how the data was made and why, see [NOTES.md](NOTES.md).

**Data folder:** `<paste the Google Drive folder link here>`

The folder contains `wesper-sv-n2w.tar` (3.8 GB): 37,153 Swedish Common Voice clips, each with
its Normal2Whisper pseudo-whisper, already cleaned up and normalized.

## What you need

- **The data folder in your My Drive.**
- **About 1.5 GB of free Drive space per training run,** for checkpoints.
- **The notebook from this repo:** `colab/wesper_sv_encoder_finetune.ipynb`.
- **A Colab GPU.** Any GPU works; the notebook picks the right precision for it. Faster GPUs
  (Colab Pro) finish sooner. If a session ends before training finishes, that's fine: you can
  resume.

## 1. Find the data folder's path

Colab sees your My Drive as `/content/drive/MyDrive`. So if the data folder is called
`wesper-sv` and sits at the top of My Drive, its path is `/content/drive/MyDrive/wesper-sv`.
That's the notebook's default, and you won't need to change anything.

If the folder has another name or location, work out its path the same way. For example,
`My Drive › projects › wesper-sv` becomes `/content/drive/MyDrive/projects/wesper-sv`.

## 2. Open the notebook in Colab

Go to [colab.research.google.com](https://colab.research.google.com), choose **File → Upload
notebook**, and select `colab/wesper_sv_encoder_finetune.ipynb` from this repo.

## 3. Choose a GPU

**Runtime → Change runtime type**, pick a GPU (for example T4), then **Save**.

## 4. Check the settings

The first code cell, **Settings**, holds everything you might change. Usually that's at most:

- **`DRIVE_DIR`:** the data folder's path from step 1. Checkpoints are saved inside it, in
  `runs/<RUN_NAME>/`.
- **`RUN_NAME`:** the name of this training run. Keep it the same to resume a run; change it to
  start a fresh one.

The rest have sensible defaults:

| Setting | Default | When to change it |
|---|---|---|
| `MAX_STEPS` | 10000 | Train longer or shorter. |
| `BATCH_SIZE` | 16 | Lower it to 8 if you get a CUDA "out of memory" error. |
| `CROP_SECONDS` | 3.0 | Seconds of audio per training example. Lowering it also saves memory. |
| `LEARNING_RATE` | 3e-5 | Leave it unless training is unstable. |
| `EVAL_EVERY`, `SAVE_EVERY` | 500, 1000 | How often to validate and save a resumable checkpoint. |

## 5. Run everything

Choose **Runtime → Run all**. When Colab asks for access to Google Drive, allow it; the
notebook needs it to read the data and save checkpoints. It then does the following:

1. Fetches the WESPER code and checks the GPU. It prints, for example,
   `GPU: Tesla T4, mixed precision: torch.float16`.
2. Copies the data from Drive to the Colab machine and unpacks it. It prints the number of
   training and validation clips.
3. Downloads WESPER's original encoder (1.1 GB). It prints a baseline:
   `Original encoder, distance to normal-speech units: whisper …, normal …`.
4. Trains.

## 6. Watch training

Every 100 steps you'll see a line like this (the numbers in these examples are made up):

```
step 1200/10000  loss 0.2140  lr 2.9e-05  3.10 steps/s  ETA 47 min
```

`ETA` is the estimated time left. Every `EVAL_EVERY` steps the notebook checks 200 validation
clips from speakers it never trains on:

```
  val: whisper 0.2050 (original 0.2600), normal 0.0310  -> saved encoder_best.pt
```

- **`whisper`:** how far the encoder's output for whispered input is from what the original
  encoder produces for the matching normal speech. This is what training improves, so it should
  fall below the `original` number. Each time it reaches a new best, `encoder_best.pt` is saved.
- **`normal`:** the same for normal-speech input. It starts near 0 and should stay small. If it
  keeps growing, the encoder is getting worse at normal speech.

When training ends, a plot shows both numbers over time.

## 7. If Colab disconnects

Free sessions in particular can end before training finishes. Open the notebook again and choose
**Runtime → Run all**, with the same `RUN_NAME`. It copies the data onto the new machine and
continues from the last checkpoint (`Resuming from step …`). Checkpoints are saved every
`SAVE_EVERY` steps, so at most that many steps are repeated.

## 8. Listen to the result

The **Listen** section runs at the end, but you can also stop training early: choose
**Runtime → Interrupt execution**, then run the cells below it. It plays four validation clips:

- the normal recording
- the Normal2Whisper input
- that input converted with the **original** encoder
- that input converted with the **fine-tuned** encoder

The next cell lets you upload your own Swedish whisper (wav, flac, mp3 or m4a) and hear it
converted with both encoders.

## 9. Use the encoder on your computer

Download `encoder_best.pt` from `DRIVE_DIR/runs/<RUN_NAME>/` into
`colab/data/runs/n2w-finetune/` in this repo.

The encoder is very sensitive to loudness, and it was trained on audio normalized to −20 dBFS
speech level. The checkpoint records that level, and WESPER (`whisper_normal.py`) normalizes the
input to it automatically. A quiet microphone and a loud one give the same result. WESPER's
original encoder records no level, so it behaves exactly as before.

- **GUI:** `./client_direct_sv.sh`. Pick your microphone and output device in the dropdowns,
  then hold the button, whisper, and release. The log shows which encoder is loaded and that
  input is normalized. The **Voice** dropdown switches between WESPER's English voice and
  every trained decoder in `decoder/runs/` (see HOW_TO_TRAIN_DECODER.md), labeled with its
  vocoder.
- **A file:** `.venv/bin/python convert.py --hubert colab/data/runs/n2w-finetune/encoder_best.pt --input my_whisper.wav --output converted.wav`
- **Server mode:** start `server.py` with the same `--hubert` path.

## Troubleshooting

- **`FileNotFoundError` mentioning `wesper-sv-n2w.tar`:** `DRIVE_DIR` doesn't point at the data
  folder. In Colab, open the file browser (folder icon on the left), go to
  `drive/MyDrive/…`, right-click the data folder, choose **Copy path**, and paste it into
  `DRIVE_DIR`.
- **`No GPU found`:** do step 3, then **Runtime → Run all** again.
- **CUDA "out of memory":** set `BATCH_SIZE` to 8 (or `CROP_SECONDS` to 2.0), then **Run all**.
  It resumes where it left off.
- **Drive is full:** each run takes about 1.5 GB (`latest.pt` 1.1 GB, `encoder_best.pt`
  0.4 GB). Delete run folders under `runs/` that you no longer need.
- **Start over:** use a new `RUN_NAME`, or delete the run's folder under `runs/`.

## Status

The notebook has been tested end-to-end on CPU with a tiny synthetic dataset: training, saving,
resuming, and conversion with both encoders. It hasn't been run on a Colab GPU yet, and the
default settings are reasonable starting points rather than tuned values.

## Rebuilding the data or running the tests

You only need this if you change how the data is prepared. These commands run in this repo on
your computer, not on Colab:

```sh
# Rebuild wesper-sv-n2w.tar (~7 min). The corpus and Normal2Whisper folders are only read.
.venv/bin/python colab/prepare_data.py ../swedish-common-voice-cv-corpus-27.0-2026-09-11 \
    ../Normal2Whisper/sv-cv27-whisper16k/clips colab/data/wesper-sv-n2w.tar --export-dir colab/data/export

# Tests for the data packer and the notebook (~3 s; add WESPER_NOTEBOOK_SMOKE=1 to also run the
# whole notebook on CPU, ~40 s once WESPER's checkpoints are cached)
.venv/bin/python -m unittest discover -s colab/tests -v

# Tests for the input normalization and the GUI's device and voice selectors (~7 s)
.venv/bin/python -m unittest discover -s tests -v
```
