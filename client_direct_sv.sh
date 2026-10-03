#!/bin/bash
# WESPER GUI with the encoder fine-tuned for Swedish whispers (see HOW_TO_TRAIN.md).
# Its input is normalized to the speech level it was trained on automatically.
# Extra arguments are passed on, e.g. --sd 4 to record from input device 4.
python client_direct.py --hubert colab/data/runs/n2w-finetune/encoder_best.pt "$@"
