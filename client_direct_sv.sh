#!/bin/bash
# WESPER GUI with the encoder fine-tuned for Swedish whispers (see HOW_TO_TRAIN.md).
# Its input is normalized to the speech level it was trained on automatically.
# Once a Swedish decoder has been trained into decoder/runs/sv-narrator (see
# HOW_TO_TRAIN_DECODER.md) it speaks with that voice; until then, WESPER's English one.
# Extra arguments are passed on, e.g. --sd 4 to record from input device 4.
cd "$(dirname "$0")"  # paths below are relative to the repo
PYTHON=.venv/bin/python  # the repo's venv, so it works without activating it
[ -x "$PYTHON" ] || PYTHON=python3
DECODER=decoder/runs/sv-narrator
VOICE=()
if [ -f "$DECODER/decoder_best.pt" ]; then
    VOICE=(--fastspeech2 "$DECODER/decoder_best.pt" --preprocess_config "$DECODER/preprocess.yaml")
fi
"$PYTHON" client_direct.py --hubert colab/data/runs/n2w-finetune/encoder_best.pt "${VOICE[@]}" "$@"
