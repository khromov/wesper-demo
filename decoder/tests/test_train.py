"""Tests for decoder/train.py and the training fix in libs/FastSpeech2/model/modules.py.

The smoke test prepares a tiny fake dataset, trains a few steps on CPU, resumes, and converts
audio with the result through WESPER. It's opt-in, because it needs WESPER's checkpoints in
torch's hub cache:

    WESPER_DECODER_SMOKE=1 .venv/bin/python -m unittest discover -s decoder/tests -v
"""
import argparse
import os
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(TESTS))
sys.path.insert(0, REPO)
sys.path.insert(0, TESTS)

from decoder import train  # noqa: E402

CACHE = os.path.join(torch.hub.get_dir(), "checkpoints")
NEEDED = [os.path.join(CACHE, f) for f in ("model-layer12-450000.pt", "googletts_neutral_best.tar", "g_00205000")]


def item(n, uid="x"):
    rng = np.random.default_rng(n)
    return {"id": uid, "units": rng.standard_normal((n, 256)).astype(np.float32),
            "mel": rng.standard_normal((n, 80)).astype(np.float32),
            "pitch": rng.standard_normal(n).astype(np.float32), "energy": rng.standard_normal(n).astype(np.float32)}


class Collate(unittest.TestCase):
    def test_fastspeech2_batch_layout(self):
        batch = train.collate([item(5, "a"), item(3, "b")])
        self.assertEqual(len(batch), 12)
        ids, _, speakers, units, src_lens, max_src_len, mels, mel_lens, max_mel_len, pitch, energy, durations = batch
        self.assertEqual(ids, ["a", "b"])
        self.assertEqual(units.shape, (2, 5, 256))
        self.assertEqual(mels.shape, (2, 5, 80))
        self.assertEqual(src_lens.tolist(), [5, 3])
        self.assertEqual((max_src_len, max_mel_len), (5, 5))
        self.assertTrue(torch.equal(src_lens, mel_lens))
        self.assertEqual(speakers.tolist(), [0, 0])
        self.assertEqual(durations.tolist(), [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        self.assertTrue((units[1, 3:] == 0).all() and (pitch[1, 3:] == 0).all())


class LengthBatches(unittest.TestCase):
    def test_full_batches_cover_each_utterance_once(self):
        lengths = list(range(10, 210))
        batches = train.length_batches(lengths, 8, random.Random(0))
        flat = [i for b in batches for i in b]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertTrue(all(len(b) == 8 for b in batches))
        self.assertEqual(len(flat), 200)

    def test_groups_similar_lengths(self):
        rng = np.random.default_rng(0)
        lengths = rng.integers(50, 600, 400).tolist()
        spread = [max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in train.length_batches(lengths, 8, random.Random(0))]
        self.assertLess(np.median(spread), 60)

    def test_small_dataset_still_gives_batches(self):
        self.assertEqual(sorted(i for b in train.length_batches([5, 6, 7], 8, random.Random(0)) for i in b), [0, 1, 2])


class Configs(unittest.TestCase):
    def test_points_wesper_at_the_run_folder(self):
        pre, model, cfg = train.configs(os.path.join(REPO, "decoder", "runs", "x"))
        self.assertEqual(pre["path"]["preprocessed_path"], os.path.join("decoder", "runs", "x"))
        self.assertEqual(pre["preprocessing"]["pitch"]["feature"], "phoneme_level")
        self.assertTrue(model["soft_unit"])
        self.assertIn("grad_clip_thresh", cfg["optimizer"])


class DurationPredictorFix(unittest.TestCase):
    """WESPER's variance adaptor skipped the duration predictor whenever durations were given,
    i.e. always in training, so the repo's loss couldn't run. It must now return predictions."""

    def test_training_forward_predicts_durations(self):
        cwd = os.getcwd()
        os.chdir(REPO)
        try:
            pre, model_config, _ = train.configs(os.path.join(REPO, "preprocessed_data", "googletts"))
            import utils.tools
            import model.modules
            from model import FastSpeech2, FastSpeech2Loss
            utils.tools.device = model.modules.device = "cpu"
            net = FastSpeech2(pre, model_config)
            batch = train.collate([item(7), item(4)])
            output = net(*batch[2:])
        finally:
            os.chdir(cwd)
        self.assertIsNotNone(output[4])
        self.assertEqual(tuple(output[4].shape), (2, 7))
        losses = FastSpeech2Loss(pre, model_config)(batch, output)
        self.assertTrue(all(torch.isfinite(l) for l in losses))


@unittest.skipUnless(os.environ.get("WESPER_DECODER_SMOKE"), "set WESPER_DECODER_SMOKE=1 to train on fake data")
class Smoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all(os.path.exists(p) for p in NEEDED):
            raise unittest.SkipTest("WESPER's checkpoints aren't cached yet")
        from test_prepare_data import make_recordings
        cls.tmp = tempfile.TemporaryDirectory()
        book, cls.data, cls.run_dir = (os.path.join(cls.tmp.name, d) for d in ("book", "data", "run"))
        make_recordings(book)
        subprocess.run([sys.executable, os.path.join(REPO, "decoder", "prepare_data.py"), book, cls.data,
                        "--val-recordings", "1"], check=True, capture_output=True, timeout=600)
        cls.first = cls.train_steps(4)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def train_steps(cls, steps):
        return subprocess.run([sys.executable, os.path.join(REPO, "decoder", "train.py"), cls.data, cls.run_dir,
                               "--steps", str(steps), "--batch-size", "2", "--eval-every", "2", "--save-every", "2",
                               "--samples", "1", "--workers", "0", "--device", "cpu"],
                              capture_output=True, text=True, timeout=900)

    def test_trains_and_writes_a_run_folder(self):
        self.assertEqual(self.first.returncode, 0, self.first.stderr[-3000:])
        self.assertIn("done at step 4", self.first.stdout)
        for name in ("decoder_best.pt", "latest.pt", "preprocess.yaml", "stats.json", "history.json"):
            self.assertTrue(os.path.exists(os.path.join(self.run_dir, name)), name)
        for folder in ("reference", "vocoded-target", "step_000002", "step_000004"):
            self.assertEqual(len(os.listdir(os.path.join(self.run_dir, "samples", folder))), 1, folder)

    def test_resumes(self):
        result = self.train_steps(6)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        self.assertIn("resuming from step 4", result.stdout)
        self.assertIn("done at step 6", result.stdout)

    def test_wesper_converts_with_the_new_decoder(self):
        # Exactly how the GUI loads a decoder: --fastspeech2 and --preprocess_config.
        cwd = os.getcwd()
        os.chdir(REPO)
        try:
            import soundfile as sf
            import whisper_normal as wn
            rel = os.path.relpath(self.run_dir, REPO)
            w2n = wn.MyWhisper2Normal(argparse.Namespace(
                preprocess_config=f"{rel}/preprocess.yaml", model_config="config/my_model16000.yaml", device="cpu",
                hubert=f"{train.RELEASE}/model-layer12-450000.pt", fastspeech2=f"{rel}/decoder_best.pt",
                hifigan=train.VOCODER))
            x, _ = sf.read(os.path.join(REPO, "sample_whisper.wav"), dtype="float32")
            out, _ = w2n.convert(x)
        finally:
            os.chdir(cwd)
        self.assertGreater(len(out), 0.5 * len(x))
        self.assertLess(len(out), 1.5 * len(x))


if __name__ == "__main__":
    unittest.main()
