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

## Optional: BigVGAN instead of HiFi-GAN

NVIDIA's BigVGAN (22.05 kHz) sounds clearly better than WESPER's HiFi-GAN, but it needs a decoder
trained for it. Prepare and train into separate folders, so both versions can be compared:

```sh
python decoder/prepare_data.py swe-audiobook/book1 decoder/data/sv-narrator-bigvgan22k --vocoder bigvgan22k --device cuda
python decoder/train.py decoder/data/sv-narrator-bigvgan22k decoder/runs/sv-narrator-bigvgan22k --device cuda
```

- **Speed:** each training step takes about 1.7× as long, because the decoder predicts 86 mel
  frames per second instead of 50.
- **Vocoder:** the run records its vocoder, so WESPER loads BigVGAN automatically:
  `DECODER=decoder/runs/sv-narrator-bigvgan22k ./client_direct_sv.sh`.
- **Checkpoint:** BigVGAN's 449 MB checkpoint downloads on first use.
- **CPU:** BigVGAN is slow on a CPU (about 1.4× real time on a MacBook), so the GUI responds
  more slowly than with HiFi-GAN.
- **Web app:** `web/export_models.py` adds this run as a third voice, *Swedish narrator
  (BigVGAN)*, when the folder exists (see web/README.md).

## Optional: fine-tune BigVGAN on the decoder

The decoder's spectrograms are smoother than real ones, and BigVGAN turns the missing detail
into an electric buzz (it's not in `samples/vocoded-target/`). Fine-tuning BigVGAN on the
finished decoder's own output teaches it to make her recordings from exactly those spectrograms.
It needs the BigVGAN run above, and one more package: `pip install --no-deps nnAudio` (pure
Python; `--no-deps` leaves your torch alone). torchaudio isn't needed.

```sh
# The utterances at 22.05 kHz, cut exactly as prepare_data.py cut them (~2 min, 1.9 GB).
python decoder/export_vocoder_audio.py swe-audiobook/book1 decoder/data/sv-narrator-bigvgan22k decoder/data/sv-narrator-bigvgan22k-audio

python decoder/finetune_vocoder.py decoder/data/sv-narrator-bigvgan22k decoder/data/sv-narrator-bigvgan22k-audio \
    decoder/runs/sv-narrator-bigvgan22k decoder/runs/sv-narrator-bigvgan22k-vft --device cuda
```

- **Pairs:** the decoder computes each utterance's spectrogram with the recording's own pitch
  and energy, as in its training. Its output lines up with the recording frame by frame, so
  BigVGAN can learn to turn it into exactly that recording.
- **Starting point:** NVIDIA's generator, discriminators and optimizer states (a 1.4 GB
  download), with NVIDIA's losses.
- **Validation:** the vocoder turns the decoder's spectrograms of the validation utterances into
  audio, and it's compared with her recordings. Step 0 is the original vocoder; lower is better.
  - `flatness gap`: how much more noise-like than her voice the output is, in 0.5–2, 2–4 and
    4–8 kHz. That's the buzz. `bigvgan_generator.pt` keeps the checkpoint with the smallest gap.
  - `mel`: the average spectrogram difference. It's only logged: it rewards smooth output, so it
    gets slightly worse while the buzz goes away.
- **Samples:** `samples/step_NNNNNN/` holds the decoder's output as WESPER makes it (pitch
  predicted) through the vocoder so far. `step_000000/` is the original vocoder, and
  `reference/` the narrator.
- **Use it:** the output folder is a complete decoder run with its own vocoder:
  `DECODER=decoder/runs/sv-narrator-bigvgan22k-vft ./client_direct_sv.sh`. Its
  `preprocess.yaml` names the vocoder (`checkpoint: bigvgan_generator.pt`), and WESPER and the
  web export load it from there.
- **Another decoder checkpoint:** `--decoder-checkpoint RUN/checkpoints/step_030000.pt` uses that
  one instead of the run's `decoder_best.pt` (see `--keep-checkpoints` below).
- **Speed first:** `--benchmark 50` times 50 steps, prints s/step and the time for `--steps`, and
  exits without saving the vocoder.
- **Stopping and resuming:** as for the decoder. Out of GPU memory: lower `--batch-size` or
  `--segment-frames`.

## Optional: fine-tune a finished decoder's postnet

The postnet is FastSpeech2's last layer, meant to add detail to the mel. In every trained decoder
here it adds almost nothing: its last BatchNorm's scale is ~0.01 (the others' ~4). A likely reason
is the 50% dropout on its output in training, which makes any correction noisy. These options
give it a fresh start on a finished decoder, training only the postnet:

```sh
python decoder/train.py decoder/data/sv-narrator-bigvgan22k decoder/runs/sv-narrator-bigvgan22k-postnet --device cuda \
    --init decoder/runs/sv-narrator-bigvgan22k/decoder_step30000.pt --reset-postnet --postnet-final-dropout 0 \
    --train-only postnet --lr 1e-4 --steps 3000 --eval-every 500 --save-every 500 --keep-checkpoints
```

- `--reset-postnet` resets its last BatchNorm (`--reset-postnet all`: the whole postnet).
  `--postnet-final-dropout 0` removes the dropout on its output; leave it out to compare.
- `--lr 1e-4` warms up over `--warmup` (200) steps, then decays to a tenth; without it, FastSpeech2's
  schedule climbs to 1e-3.
- Every validation prints the buzz measures next to the losses: `flatness` of the samples in
  0.5–2/2–4/4–8 kHz against `vocoded target`'s, and `sharpness` of the mels against the real ones'.
  Better is closer to those, with `val` mel and pitch about the same.
- `--keep-checkpoints` keeps each validation's weights in `checkpoints/step_NNNNNN.pt`, for
  `finetune_vocoder.py --decoder-checkpoint`.
- `--benchmark 50` times 50 steps and exits without saving.

## On the training server (Radeon 760M, ROCm)

MIOpen's convolution kernels hang this GPU, so every GPU command goes through `decoder/rocm.py`,
which turns MIOpen off and runs the script with the same arguments (`decoder/train_rocm.py` does
the same for `train.py` only). Use `.venv-rocm/bin/python`. The server has no ffmpeg:
`prepare_data.py` and `export_vocoder_audio.py` run elsewhere, and their output is copied over.

```sh
cd ~/wesper-demo

# Once: the vocoder fine-tuning's one new package. --no-deps keeps the ROCm build of torch.
~/.local/bin/uv pip install --python .venv-rocm/bin/python --no-deps nnAudio

# Fine-tune BigVGAN on the decoder (decoder/data/sv-narrator-bigvgan22k-audio comes from export_vocoder_audio.py).
# First the speed: it computes and caches the decoder's mels (once), then times 50 steps and saves no vocoder.
.venv-rocm/bin/python decoder/rocm.py decoder/finetune_vocoder.py \
    decoder/data/sv-narrator-bigvgan22k decoder/data/sv-narrator-bigvgan22k-audio \
    decoder/runs/sv-narrator-bigvgan22k decoder/runs/sv-narrator-bigvgan22k-vft \
    --decoder-checkpoint decoder/runs/sv-narrator-bigvgan22k/decoder_step30000.pt --device cuda --benchmark 50

# Then the run: the same command without --benchmark, in the background. Re-run it to resume.
nohup .venv-rocm/bin/python decoder/rocm.py decoder/finetune_vocoder.py \
    decoder/data/sv-narrator-bigvgan22k decoder/data/sv-narrator-bigvgan22k-audio \
    decoder/runs/sv-narrator-bigvgan22k decoder/runs/sv-narrator-bigvgan22k-vft \
    --decoder-checkpoint decoder/runs/sv-narrator-bigvgan22k/decoder_step30000.pt --device cuda \
    > decoder/runs/sv-narrator-bigvgan22k-vft.log 2>&1 &
```

The postnet fine-tuning above runs the same way: `.venv-rocm/bin/python decoder/rocm.py
decoder/train.py …` with its options (try `--benchmark 50` first). Fine-tune the vocoder on its
result afterwards: the vocoder learns the decoder it's given.

## Checking the setup

```sh
WESPER_DECODER_SMOKE=1 python -m unittest discover -s decoder/tests
```

This runs the whole pipeline on a tiny fake dataset in about a minute.
