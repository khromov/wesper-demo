# How to train the Swedish voice (decoder)

This trains WESPER's decoder on the audiobook narrator (`swe-audiobook/book1/`, 15 h), so that
WESPER speaks Swedish with her voice instead of its English one. No transcripts are needed.
Background, and how this compares with WESPER's own training: [NOTES.md](NOTES.md).

Run everything from the repo root on the training computer.

## 1. Environment

The Python environment you trained the encoder with works. For step 2 it also needs `pyworld`,
and `ffmpeg` on the PATH:

```sh
pip install pyworld "setuptools<81"
```

## 2. Prepare the data

```sh
python decoder/prepare_data.py swe-audiobook/book1 decoder/data/sv-narrator --device cuda
```

If you copied the repo with `decoder/data/sv-narrator/` already prepared, this takes seconds: it
skips finished chapters and only rebuilds the list and statistics. Otherwise it processes all 187
chapters, which took about 1.5 hours on a laptop CPU. It ends with a summary like
`train: … utterances, … h`.

## 3. Train

```sh
python decoder/train.py decoder/data/sv-narrator decoder/runs/sv-narrator --device cuda
```

- **Starting point:** WESPER's Google TTS decoder. Training runs 30,000 steps and prints
  progress and an ETA every 100 steps.
- **Every 2,000 steps:** it saves `decoder_best.pt` if the validation loss improved, and writes
  audio to `decoder/runs/sv-narrator/samples/`:
  - `reference/`: the narrator.
  - `vocoded-target/`: the best this vocoder can sound.
  - `step_NNNNNN/`: the decoder so far.
- **Stopping and resuming:** stop whenever the newest samples sound good. If it stops, run the
  same command again to resume. To train longer, re-run with a higher `--steps`.
- **Out of GPU memory:** add `--batch-size 8`.

## 4. Use it

Copy `decoder/runs/sv-narrator/` back into this repo (only `decoder_best.pt`, `preprocess.yaml`
and `stats.json` are needed), then run `./client_direct_sv.sh`. The GUI picks up the new voice
automatically; its log shows `decoder: decoder_best.pt`.

## Checking the setup

```sh
WESPER_DECODER_SMOKE=1 python -m unittest discover -s decoder/tests
```

This runs the whole pipeline on a tiny fake dataset in about a minute.
