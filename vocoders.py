"""The vocoders WESPER can use, and the mel spectrogram each one expects.

A decoder is trained for one vocoder: decoder/prepare_data.py --vocoder computes the mel targets
with that vocoder's settings, and decoder/train.py records the vocoder's name in the run's
preprocess.yaml (vocoder: {name: ...}). whisper_normal.py reads it and loads the matching
vocoder. Decoders without that entry, like WESPER's released ones, use WESPER's own HiFi-GAN.

  hifigan16k   WESPER's 16 kHz HiFi-GAN (hifigan/, checkpoint g_00205000). 80 mels up to 8 kHz,
               hop 320 = 20 ms: exactly one mel frame per speech unit.
  bigvgan22k   NVIDIA's BigVGAN v2, bigvgan_v2_22khz_80band_fmax8k_256x (code in libs/bigvgan).
               22.05 kHz audio, 80 mels up to 8 kHz like WESPER's, hop 256 = 11.6 ms. The units
               are interpolated to its frame rate.
  bigvgan:<m>  any other BigVGAN v2 model published at https://huggingface.co/nvidia, e.g.
               bigvgan:bigvgan_v2_24khz_100band_256x.

All of them compute mels the same way (HiFi-GAN's and BigVGAN's meldataset.mel_spectrogram),
each with its own settings. Checkpoints download into torch's hub cache on first use.
"""
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.abspath(__file__))
UNIT_SAMPLE_RATE, UNIT_HOP = 16000, 320  # the encoder's units: one per 320 samples at 16 kHz
UNIT_SECONDS = UNIT_HOP / UNIT_SAMPLE_RATE
DEFAULT = "hifigan16k"
ALIASES = {"bigvgan22k": "bigvgan:bigvgan_v2_22khz_80band_fmax8k_256x"}
WESPER_RELEASE = "https://github.com/rkmt/wesper-demo/releases/download/v0.1"
NVIDIA = "https://huggingface.co/nvidia"


@dataclass(frozen=True)
class Spec:
    """A vocoder's audio and mel settings, and where its checkpoint is."""
    name: str
    sample_rate: int
    n_fft: int
    hop: int
    win: int
    n_mels: int
    fmin: float
    fmax: Optional[float]  # None: half the sample rate
    checkpoint: str

    @property
    def frame_seconds(self):
        return self.hop / self.sample_rate

    @property
    def one_frame_per_unit(self):
        return self.sample_rate == 16000 and self.hop == 320


HIFIGAN16K = Spec("hifigan16k", 16000, 1024, 320, 1024, 80, 0, 8000, f"{WESPER_RELEASE}/g_00205000")


def fetch(url, folder=""):
    """Local path of a file, downloaded into torch's hub cache the first time."""
    directory = os.path.join(torch.hub.get_dir(), "checkpoints", folder)
    path = os.path.join(directory, os.path.basename(url))
    if not os.path.exists(path):
        os.makedirs(directory, exist_ok=True)
        torch.hub.download_url_to_file(url, path)
    return path


def _bigvgan_model(name):
    return ALIASES.get(name, name).split(":", 1)[1]


def _bigvgan_config(name):
    model = _bigvgan_model(name)
    with open(fetch(f"{NVIDIA}/{model}/resolve/main/config.json", os.path.join("bigvgan", model))) as f:
        return json.load(f)


def spec(name=DEFAULT):
    """The settings of a vocoder, by name (see the module docstring)."""
    if name == "hifigan16k":
        return HIFIGAN16K
    if not ALIASES.get(name, name).startswith("bigvgan:"):
        raise ValueError(f"unknown vocoder {name!r}: use hifigan16k, bigvgan22k or bigvgan:<model>")
    c = _bigvgan_config(name)
    model = _bigvgan_model(name)
    return Spec(name, c["sampling_rate"], c["n_fft"], c["hop_size"], c["win_size"], c["num_mels"], c["fmin"],
                c["fmax"], f"{NVIDIA}/{model}/resolve/main/bigvgan_generator.pt")


_MEL_BASIS = {}


def mel_energy(x, s=HIFIGAN16K):
    """Log-mel spectrogram (n_mels x T) and energy (T) of float audio at s.sample_rate, T = len(x) // s.hop.

    HiFi-GAN's and BigVGAN's meldataset.mel_spectrogram: reflect-pad (n_fft - hop) / 2, uncentered
    STFT, magnitude, mel filterbank, natural log clamped at 1e-5. Frame t is centered on sample
    t * hop + hop / 2. Energy is the L2 norm of the frame's magnitude spectrum, as in FastSpeech2.
    """
    import librosa
    if s not in _MEL_BASIS:
        _MEL_BASIS[s] = torch.from_numpy(
            librosa.filters.mel(sr=s.sample_rate, n_fft=s.n_fft, n_mels=s.n_mels, fmin=s.fmin, fmax=s.fmax)).float()
    y = torch.from_numpy(np.asarray(x, dtype=np.float32))[None, None]
    y = F.pad(y, ((s.n_fft - s.hop) // 2, (s.n_fft - s.hop) // 2), mode="reflect")[0, 0]
    spec_ = torch.stft(y, s.n_fft, s.hop, s.win, torch.hann_window(s.win), center=False, return_complex=True)
    mag = torch.sqrt(spec_.real ** 2 + spec_.imag ** 2 + 1e-9)
    mel = torch.log(torch.clamp(_MEL_BASIS[s] @ mag, min=1e-5))
    return mel.numpy(), torch.linalg.norm(mag, dim=0).numpy()


def n_frames(n_units, s):
    """How many of the vocoder's frames the decoder makes from n_units speech units: as many as
    fit in their duration. Integer arithmetic, so the browser export computes exactly the same."""
    if s.one_frame_per_unit:
        return n_units
    return n_units * UNIT_HOP * s.sample_rate // (UNIT_SAMPLE_RATE * s.hop)


def units_to_frames(units, frames, s):
    """Units (T x D, one per 20 ms) at the vocoder's frame rate: `frames` x D.

    Linear interpolation at each frame's center: unit i is centered at (i + 0.5) * 20 ms and
    frame j at (j + 0.5) * hop / sample_rate. For hifigan16k that's the units themselves.
    Works on numpy arrays and torch tensors (on any device).
    """
    if s.one_frame_per_unit:
        return units[:frames]
    u = units if torch.is_tensor(units) else torch.from_numpy(np.asarray(units))
    t = u.shape[0]
    pos = ((torch.arange(frames, dtype=torch.float64, device=u.device) + 0.5) * s.frame_seconds / UNIT_SECONDS
           - 0.5).clamp(0, t - 1)
    lo = pos.floor().long()
    hi = (lo + 1).clamp(max=t - 1)
    w = (pos - lo).to(u.dtype)[:, None]
    out = u[lo] * (1 - w) + u[hi] * w
    return out if torch.is_tensor(units) else out.numpy()


def load(s, device="cpu", checkpoint=None):
    """The vocoder, ready for inference: mel (B x n_mels x T) -> audio (B x 1 x T * hop)."""
    if s.name == "hifigan16k":
        import hifigan
        with open(os.path.join(REPO, "hifigan", "my_config_v1_16000.json")) as f:
            generator = hifigan.Generator(hifigan.AttrDict(json.load(f)))
        path = checkpoint or s.checkpoint
    else:
        from libs.bigvgan.bigvgan import BigVGAN
        from libs.bigvgan.env import AttrDict
        generator = BigVGAN(AttrDict(_bigvgan_config(s.name)), use_cuda_kernel=False)
        path = checkpoint or s.checkpoint
    if path.startswith("http"):
        path = fetch(path, os.path.join("bigvgan", _bigvgan_model(s.name)) if s.name != "hifigan16k" else "")
    generator.load_state_dict(torch.load(path, map_location="cpu")["generator"])
    generator.eval()
    generator.remove_weight_norm()
    return generator.to(device)


def synthesize(vocoder, mel):
    """Float audio (numpy, at the vocoder's sample rate) from a mel spectrogram (n_mels x T)."""
    device = next(vocoder.parameters()).device
    with torch.no_grad():
        return vocoder(torch.as_tensor(mel, dtype=torch.float32, device=device)[None]).squeeze().float().cpu().numpy()
