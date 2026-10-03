#!/bin/bash
# Start the WESPER GUI with the encoder fine-tuned for Swedish whispers, using the repo's .venv.
# Works from any directory. Extra arguments are passed on, e.g. ./startgui.sh --sd 4
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "No .venv in $PWD. Create it with: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi
ENCODER=colab/data/runs/n2w-finetune/encoder_best.pt
if [ ! -f "$ENCODER" ]; then
  echo "Swedish encoder not found at $ENCODER (see HOW_TO_TRAIN.md)" >&2
  exit 1
fi

export PATH="$PWD/.venv/bin:$PATH"
exec bash client_direct_sv.sh "$@"
