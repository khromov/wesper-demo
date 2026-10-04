"""Tests for vocoders.py: the vocoder settings, their mel spectrograms, and units-to-frames mapping.

BigVGAN's settings come from its config.json (a small download, cached). The tests that run
BigVGAN itself need its 449 MB checkpoint and are opt-in:

    WESPER_BIGVGAN=1 .venv/bin/python -m unittest discover -s tests -p test_vocoders.py -v
"""
import os
import sys
import unittest

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import vocoders  # noqa: E402

rng = np.random.default_rng(0)


def noise(seconds, sample_rate, dbfs=-20):
    return (rng.standard_normal(int(seconds * sample_rate)) * 10 ** (dbfs / 20)).astype(np.float32)


def bigvgan_mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax):
    """BigVGAN's meldataset.mel_spectrogram (NVIDIA/BigVGAN, main), line for line, as the reference."""
    from librosa.filters import mel as librosa_mel_fn
    mel_basis = torch.from_numpy(librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)).float()
    hann_window = torch.hann_window(win_size)
    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=False,
                      pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    return torch.log(torch.clamp(torch.matmul(mel_basis, spec), min=1e-5))


class Specs(unittest.TestCase):
    def test_hifigan16k_is_wespers_vocoder(self):
        s = vocoders.spec("hifigan16k")
        self.assertEqual((s.sample_rate, s.n_fft, s.hop, s.win, s.n_mels, s.fmin, s.fmax), (16000, 1024, 320, 1024, 80, 0, 8000))
        self.assertTrue(s.one_frame_per_unit)
        self.assertEqual(vocoders.spec(), s)

    def test_bigvgan22k_settings_come_from_its_config(self):
        s = vocoders.spec("bigvgan22k")
        self.assertEqual((s.sample_rate, s.n_fft, s.hop, s.win, s.n_mels, s.fmin, s.fmax), (22050, 1024, 256, 1024, 80, 0, 8000))
        self.assertFalse(s.one_frame_per_unit)
        self.assertTrue(s.checkpoint.endswith("bigvgan_v2_22khz_80band_fmax8k_256x/resolve/main/bigvgan_generator.pt"))

    def test_unknown_name_is_refused(self):
        with self.assertRaises(ValueError):
            vocoders.spec("wavenet")


class MelEnergy(unittest.TestCase):
    def test_matches_bigvgans_own_mel(self):
        s = vocoders.spec("bigvgan22k")
        x = noise(1.0, s.sample_rate)
        mel, energy = vocoders.mel_energy(x, s)
        expected = bigvgan_mel_spectrogram(torch.from_numpy(x)[None], s.n_fft, s.n_mels, s.sample_rate, s.hop, s.win,
                                           s.fmin, s.fmax)[0].numpy()
        np.testing.assert_allclose(mel, expected, atol=1e-5)
        self.assertEqual(energy.shape, (len(x) // s.hop,))

    def test_frames_are_centered_mid_hop(self):
        for s in (vocoders.spec("hifigan16k"), vocoders.spec("bigvgan22k")):
            for t in (7, 30):
                x = np.zeros(s.sample_rate, np.float32)
                x[t * s.hop + s.hop // 2] = 1.0
                _, energy = vocoders.mel_energy(x, s)
                self.assertEqual(int(np.argmax(energy)), t, s.name)


class UnitsToFrames(unittest.TestCase):
    def test_frame_counts(self):
        self.assertEqual(vocoders.n_frames(100, vocoders.spec("hifigan16k")), 100)
        self.assertEqual(vocoders.n_frames(100, vocoders.spec("bigvgan22k")), 172)  # 2 s at 22050 / 256

    def test_hifigan16k_uses_the_units_as_they_are(self):
        u = rng.standard_normal((50, 256)).astype(np.float32)
        np.testing.assert_array_equal(vocoders.units_to_frames(u, 50, vocoders.spec("hifigan16k")), u)

    def test_interpolates_at_frame_centers(self):
        s = vocoders.spec("bigvgan22k")
        ramp = np.repeat(np.arange(50, dtype=np.float32)[:, None], 4, axis=1)  # unit i has value i
        out = vocoders.units_to_frames(ramp, 86, s)
        expected = np.clip((np.arange(86) + 0.5) * s.frame_seconds / vocoders.UNIT_SECONDS - 0.5, 0, 49)
        self.assertEqual(out.shape, (86, 4))
        np.testing.assert_allclose(out[:, 0], expected, atol=1e-5)

    def test_constant_units_stay_constant_and_tensors_stay_tensors(self):
        u = torch.full((20, 8), 3.0)
        out = vocoders.units_to_frames(u, 34, vocoders.spec("bigvgan22k"))
        self.assertTrue(torch.is_tensor(out))
        self.assertTrue(torch.allclose(out, torch.full((34, 8), 3.0)))


class RunCheckpoint(unittest.TestCase):
    def test_a_fine_tuned_run_names_its_vocoder_relative_to_its_folder(self):
        pre = {"vocoder": {"name": "bigvgan22k", "checkpoint": "bigvgan_generator.pt"}}
        self.assertEqual(vocoders.run_checkpoint(pre, "/runs/x"), os.path.join("/runs/x", "bigvgan_generator.pt"))

    def test_other_runs_use_the_released_vocoder(self):
        for pre in ({}, {"vocoder": {"name": "bigvgan22k"}}, {"vocoder": {"name": "hifigan16k"}}):
            self.assertIsNone(vocoders.run_checkpoint(pre, "/runs/x"))

    def test_bigvgan_spec_reads_the_config(self):
        c = {"sampling_rate": 22050, "n_fft": 1024, "hop_size": 256, "win_size": 1024, "num_mels": 80, "fmin": 0, "fmax": 8000}
        s = vocoders.bigvgan_spec("bigvgan22k", c)
        self.assertEqual((s.sample_rate, s.hop, s.n_mels, s.fmax), (22050, 256, 80, 8000))
        self.assertTrue(s.checkpoint.endswith("/bigvgan_v2_22khz_80band_fmax8k_256x/resolve/main/bigvgan_generator.pt"))


BIGVGAN_CACHE = os.path.join(torch.hub.get_dir(), "checkpoints", "bigvgan", "bigvgan_v2_22khz_80band_fmax8k_256x",
                             "bigvgan_generator.pt")


@unittest.skipUnless(os.environ.get("WESPER_BIGVGAN"), "set WESPER_BIGVGAN=1 to run BigVGAN (449 MB download)")
class BigVGAN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = vocoders.spec("bigvgan22k")
        cls.vocoder = vocoders.load(cls.spec)

    def test_synthesizes_hop_samples_per_frame(self):
        mel, _ = vocoders.mel_energy(noise(1.0, self.spec.sample_rate), self.spec)
        y = vocoders.synthesize(self.vocoder, mel)
        self.assertEqual(len(y), mel.shape[1] * self.spec.hop)
        self.assertLessEqual(np.abs(y).max(), 1.0)

    def test_round_trip_keeps_the_mel(self):
        t = np.arange(self.spec.sample_rate) / self.spec.sample_rate
        x = (0.1 * sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 20))).astype(np.float32)  # a voiced tone
        mel, _ = vocoders.mel_energy(x, self.spec)
        again, _ = vocoders.mel_energy(vocoders.synthesize(self.vocoder, mel), self.spec)
        self.assertLess(np.abs(mel - again).mean(), 0.3)


if __name__ == "__main__":
    unittest.main()
