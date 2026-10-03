"""Tests for input-level normalization in whisper_normal.py (used by the GUI, server and convert.py).

Encoders fine-tuned on loudness-normalized audio store the target level in their checkpoint, and
MyWhisper2Normal normalizes their input to it. WESPER's original encoder stores none, so its
input must pass through unchanged.

The model tests need WESPER's checkpoints in torch's hub cache (any earlier conversion run puts
them there) and are skipped otherwise.

    .venv/bin/python -m unittest discover -s tests -v
"""
import argparse
import os
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(torch.hub.get_dir(), "checkpoints")
RELEASE = "https://github.com/rkmt/wesper-demo/releases/download/v0.1"
ORIGINAL = f"{RELEASE}/model-layer12-450000.pt"
CACHED = [os.path.join(CACHE, f) for f in ("model-layer12-450000.pt", "googletts_neutral_best.tar", "g_00205000")]

_cwd = os.getcwd()
os.chdir(REPO)  # whisper_normal loads libs and configs by paths relative to the repo
sys.path.insert(0, REPO)
import whisper_normal as wn  # noqa: E402
from colab.prepare_data import speech_dbfs  # noqa: E402


def tearDownModule():
    os.chdir(_cwd)


def noise(seconds, dbfs, seed=0):
    return (np.random.default_rng(seed).standard_normal(int(seconds * 16000)) * 10 ** (dbfs / 20)).astype(np.float32)


class NormalizeLevel(unittest.TestCase):
    def test_reaches_target(self):
        for level in (-50, -20, -6):
            self.assertAlmostEqual(speech_dbfs(wn.normalize_level(noise(1, level), -20.0)), -20.0, delta=0.05)

    def test_caps_gain(self):
        self.assertAlmostEqual(speech_dbfs(wn.normalize_level(noise(1, -75), -20.0, 40.0)), -35.0, delta=0.5)

    def test_keeps_shape_and_accepts_tensors(self):
        x = noise(1, -40)
        for wav in (x[None, :], torch.from_numpy(x)[None, :]):
            y = wn.normalize_level(wav, -20.0)
            self.assertEqual(y.shape, (1, len(x)))
            self.assertAlmostEqual(speech_dbfs(y.reshape(-1)), -20.0, delta=0.05)

    def test_does_not_clip(self):
        x = noise(1, -40)
        x[8000] = 0.5  # a peak far above the speech level
        y = wn.normalize_level(x, -20.0)
        self.assertGreater(np.abs(y).max(), 1.0)

    def test_silence_stays_silent(self):
        self.assertEqual(np.abs(wn.normalize_level(np.zeros(16000, np.float32), -20.0)).max(), 0.0)


@unittest.skipUnless(all(os.path.exists(p) for p in CACHED), "WESPER's checkpoints aren't cached yet")
class Conversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        # A "fine-tuned" checkpoint in the notebook's format: the original weights plus a target level.
        state = torch.load(CACHED[0], map_location="cpu")["hubert"]
        cls.finetuned = os.path.join(cls.tmp.name, "encoder_best.pt")
        torch.save({"hubert": state, "step": 1, "config": {"TARGET_DBFS": -20.0, "MAX_GAIN_DB": 40.0}}, cls.finetuned)
        cls.whisper, _ = sf.read(os.path.join(REPO, "sample_whisper.wav"), dtype="float32")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def w2n(self, hubert):
        return wn.MyWhisper2Normal(argparse.Namespace(
            preprocess_config="config/my_preprocess16k_LJ.yaml", model_config="config/my_model16000.yaml",
            device="cpu", hubert=hubert, fastspeech2=f"{RELEASE}/googletts_neutral_best.tar",
            hifigan=f"{RELEASE}/g_00205000"))

    def test_original_encoder_input_is_unchanged(self):
        w2n = self.w2n(ORIGINAL)
        self.assertIsNone(w2n.target_dbfs)
        units = w2n.wav2units(torch.from_numpy(self.whisper)[None])
        direct = wn.wav2units(torch.from_numpy(self.whisper)[None], w2n.encoder, device="cpu")
        torch.testing.assert_close(units, direct)

    def test_original_encoder_output_depends_on_input_level(self):
        w2n = self.w2n(ORIGINAL)
        loud, _ = w2n.convert(self.whisper)
        quiet, _ = w2n.convert(self.whisper * 0.1)
        self.assertFalse(np.array_equal(loud, quiet))

    def test_finetuned_encoder_normalizes_input_level(self):
        w2n = self.w2n(self.finetuned)
        self.assertEqual((w2n.target_dbfs, w2n.max_gain_db), (-20.0, 40.0))
        loud, _ = w2n.convert(self.whisper)
        quiet, _ = w2n.convert(self.whisper * 0.1)  # 20 dB quieter, as from a distant microphone
        np.testing.assert_allclose(loud.astype(np.float32), quiet.astype(np.float32), atol=2)  # int16 output

    def test_finetuned_conversion_matches_normalizing_by_hand(self):
        w2n = self.w2n(self.finetuned)
        out, _ = w2n.convert(self.whisper)
        w2n.target_dbfs = None  # same encoder without automatic normalization
        by_hand, _ = w2n.convert(wn.normalize_level(self.whisper, -20.0))
        np.testing.assert_array_equal(out, by_hand)


if __name__ == "__main__":
    unittest.main()
