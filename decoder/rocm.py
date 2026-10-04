"""Run a decoder/ script on an AMD GPU whose MIOpen kernels hang, like the Radeon 760M (gfx1103):
MIOpen off (torch.backends.cudnn.enabled = False on ROCm), PyTorch's own kernels instead. The
script gets the same arguments. It generalizes the training machine's decoder/train_rocm.py.

    .venv-rocm/bin/python decoder/rocm.py decoder/train.py DATA_DIR RUN_DIR [train.py options]
    .venv-rocm/bin/python decoder/rocm.py decoder/finetune_vocoder.py DATA_DIR AUDIO_DIR DECODER_RUN OUT_DIR [options]
"""
import runpy
import sys

import torch

if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1].endswith(".py"):
        sys.exit(__doc__)
    torch.backends.cudnn.enabled = False
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
