# torchaudio.transforms.Resample's default method (sinc interpolation with a Hann window), as of
# torchaudio 2.0, so that the CQT discriminator runs without torchaudio: its builds are tied to one
# torch build, and the ROCm machine's torch has no matching one. Same kernel, same output.
# From https://github.com/pytorch/audio (torchaudio/functional/functional.py, transforms/_transforms.py),
# BSD 2-Clause License, Copyright (c) 2017 Facebook Inc. (Soumith Chintala).
import math

import torch
import torch.nn.functional as F


def sinc_resample_kernel(orig_freq, new_freq, gcd, lowpass_filter_width=6, rolloff=0.99):
    """torchaudio.functional._get_sinc_resample_kernel for sinc_interp_hann: (kernel, width)."""
    orig_freq, new_freq = int(orig_freq) // gcd, int(new_freq) // gcd
    base_freq = min(orig_freq, new_freq) * rolloff
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    idx = torch.arange(-width, width + orig_freq, dtype=torch.float64)[None, None] / orig_freq
    t = torch.arange(0, -new_freq, -1, dtype=torch.float32)[:, None, None] / new_freq + idx  # float32 here, as in torchaudio
    t *= base_freq
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    t *= math.pi
    kernels = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernels *= window * (base_freq / orig_freq)
    return kernels.to(dtype=torch.float32), width


class Resample(torch.nn.Module):
    """torchaudio.transforms.Resample(orig_freq, new_freq) with its defaults."""

    def __init__(self, orig_freq=16000, new_freq=16000):
        super().__init__()
        self.orig_freq, self.new_freq = orig_freq, new_freq
        self.gcd = math.gcd(int(orig_freq), int(new_freq))
        if orig_freq != new_freq:
            kernel, self.width = sinc_resample_kernel(orig_freq, new_freq, self.gcd)
            self.register_buffer("kernel", kernel)

    def forward(self, waveform):
        if self.orig_freq == self.new_freq:
            return waveform
        orig_freq, new_freq = int(self.orig_freq) // self.gcd, int(self.new_freq) // self.gcd
        shape = waveform.size()
        waveform = waveform.view(-1, shape[-1])
        num_wavs, length = waveform.shape
        waveform = F.pad(waveform, (self.width, self.width + orig_freq))
        resampled = F.conv1d(waveform[:, None], self.kernel, stride=orig_freq)
        resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
        resampled = resampled[..., : int(math.ceil(new_freq * length / orig_freq))]
        return resampled.view(shape[:-1] + resampled.shape[-1:])
