"""Tests for decoder/bigvgan_preview.py's mel conversion (HiFi-GAN 16 kHz mel -> BigVGAN-22k mel).

    .venv/bin/python -m unittest discover -s decoder/tests -p test_bigvgan_preview.py -v
"""
import os
import sys
import unittest

import numpy as np

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(TESTS))
sys.path.insert(0, REPO)

import vocoders  # noqa: E402
from decoder import bigvgan_preview as bp  # noqa: E402

rng = np.random.default_rng(0)
BIG = vocoders.spec("bigvgan22k")


class Conversion(unittest.TestCase):
    def test_frames_land_at_bigvgans_frame_centers(self):
        mel16 = rng.standard_normal((80, 50)).astype(np.float32)
        out = bp.to_frames(mel16, BIG, 86)
        self.assertEqual(out.shape, (80, 86))
        np.testing.assert_allclose(out, vocoders.units_to_frames(mel16.T, 86, BIG).T)

    def test_band_fit_recovers_a_known_line(self):
        a_true, c_true = rng.uniform(0.8, 1.1, 80), rng.uniform(-0.5, 0.5, 80)
        pairs = []
        for _ in range(4):
            x = rng.normal(-6, 2, (80, 300))
            pairs.append((x, a_true[:, None] * x + c_true[:, None] + rng.normal(0, 0.01, x.shape)))
        a, c = bp.fit_band_map(pairs)
        np.testing.assert_allclose(a, a_true, atol=0.01)
        np.testing.assert_allclose(c, c_true, atol=0.05)

    def test_apply_band_map(self):
        mel = np.ones((80, 5), np.float32)
        out = bp.apply_band_map(mel, np.full(80, 2.0), np.arange(80.0))
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out[:, 0], 2 + np.arange(80))

    def test_both_mels_are_aligned(self):
        t16, t22 = np.arange(16000 * 2) / 16000, np.arange(22050 * 2) / 22050
        tone = lambda t: (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        m16, m22 = bp.both_mels(tone(t16), tone(t22), BIG)
        self.assertEqual(m16.shape, m22.shape)
        self.assertEqual(m22.shape[0], 80)
        self.assertGreaterEqual(m22.shape[1], vocoders.n_frames(100, BIG) - 2)
        # The same steady tone analyzed both ways: the bands nearly agree once the levels are matched.
        self.assertLess(np.abs((m16 - m16.mean()) - (m22 - m22.mean())).mean(), 1.0)


if __name__ == "__main__":
    unittest.main()
