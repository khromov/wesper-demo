"""Tests for decoder/train.py and the training fix in libs/FastSpeech2/model/modules.py.

The smoke test prepares a tiny fake dataset, trains a few steps on CPU, resumes, and converts
audio with the result through WESPER. It's opt-in, because it needs WESPER's checkpoints in
torch's hub cache:

    WESPER_DECODER_SMOKE=1 .venv/bin/python -m unittest discover -s decoder/tests -v
"""
import argparse
import json
import os
import random
import shutil
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


class VocoderConfigs(unittest.TestCase):
    def test_bigvgan_run_config_names_the_vocoder_and_its_mel_settings(self):
        import vocoders
        pre, _, _ = train.configs(os.path.join(REPO, "decoder", "runs", "x"), vocoders.spec("bigvgan22k"))
        self.assertEqual(pre["vocoder"], {"name": "bigvgan22k"})
        p = pre["preprocessing"]
        self.assertEqual(p["audio"]["sampling_rate"], 22050)
        self.assertEqual((p["stft"]["filter_length"], p["stft"]["hop_length"], p["stft"]["win_length"]), (1024, 256, 1024))
        self.assertEqual((p["mel"]["n_mel_channels"], p["mel"]["mel_fmax"]), (80, 8000))

    def test_hifigan_run_config_keeps_wespers_settings(self):
        pre, _, _ = train.configs(os.path.join(REPO, "decoder", "runs", "x"))
        self.assertEqual(pre["vocoder"], {"name": "hifigan16k"})
        self.assertEqual((pre["preprocessing"]["audio"]["sampling_rate"], pre["preprocessing"]["stft"]["hop_length"]), (16000, 320))


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


def tiny_decoder():
    """FastSpeech2 as train.py builds it, with random weights, on the CPU."""
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        pre, model_config, _ = train.configs(os.path.join(REPO, "preprocessed_data", "googletts"))
        import utils.tools
        import model.modules
        from model import FastSpeech2
        utils.tools.device = model.modules.device = "cpu"
        torch.manual_seed(0)
        return FastSpeech2(pre, model_config)
    finally:
        os.chdir(cwd)


class FineTuning(unittest.TestCase):
    """The options for fine-tuning a finished decoder (--init, --reset-postnet, --lr, --train-only)."""

    def test_loads_any_decoder_checkpoint(self):
        net = tiny_decoder()
        sd = net.state_dict()
        with tempfile.TemporaryDirectory() as tmp:
            for name, obj in (("best.pt", {"model": sd, "step": 3, "val": {}}), ("bare.pt", sd),
                              ("latest.pt", {"model": sd, "optimizer": {}, "step": 3})):
                torch.save(obj, os.path.join(tmp, name))
                loaded = train.load_init(os.path.join(tmp, name))
                self.assertEqual(set(loaded), set(sd), name)

    def final_norm(self, net):
        return net.postnet.convolutions[-1][1]

    def test_reset_final_norm_unmutes_only_the_last_batchnorm(self):
        net = tiny_decoder()
        with torch.no_grad():
            for m in net.postnet.convolutions:
                m[1].weight.fill_(0.01)
        conv_before = net.postnet.convolutions[-1][0].conv.weight.clone()
        train.reset_postnet(net, "final-norm")
        final = self.final_norm(net)
        self.assertTrue(torch.equal(final.weight, torch.ones_like(final.weight)))
        self.assertTrue(torch.equal(final.bias, torch.zeros_like(final.bias)))
        self.assertTrue(torch.equal(final.running_var, torch.ones_like(final.running_var)))
        self.assertTrue(torch.equal(net.postnet.convolutions[-1][0].conv.weight, conv_before))
        self.assertTrue(torch.allclose(net.postnet.convolutions[0][1].weight, torch.tensor(0.01)))

    def test_reset_all_reinitializes_the_postnet_only(self):
        net = tiny_decoder()
        before = {k: v.clone() for k, v in net.state_dict().items()}
        with torch.no_grad():
            for p in net.postnet.parameters():
                p.fill_(0.5)
        train.reset_postnet(net, "all")
        after = net.state_dict()
        for k in before:
            if k.startswith("postnet.") and k.endswith("conv.weight"):
                self.assertFalse(torch.allclose(after[k], torch.tensor(0.5)), k)
            elif not k.startswith("postnet."):
                self.assertTrue(torch.equal(after[k], before[k]), k)
        self.assertTrue(torch.equal(self.final_norm(net).weight, torch.ones(80)))

    def test_postnet_final_dropout_applies_to_the_correction_in_training_only(self):
        net = tiny_decoder()
        self.assertEqual(net.postnet.final_dropout, 0.5)  # FastSpeech2's, unless --postnet-final-dropout
        x = torch.randn(2, 20, 80)
        net.postnet.final_dropout = 1.0
        net.train()
        self.assertTrue(torch.equal(net.postnet(x), torch.zeros_like(x)))
        net.eval()
        a = net.postnet(x)
        net.postnet.final_dropout = 0.5
        torch.testing.assert_close(net.postnet(x), a)
        self.assertGreater(a.abs().mean().item(), 0)

    def test_finetune_lr_warms_up_then_decays_to_a_tenth(self):
        lrs = [train.finetune_lr(s, 1e-4, 100, 1000) for s in range(1000)]
        self.assertAlmostEqual(lrs[0], 1e-6)
        self.assertAlmostEqual(lrs[99], 1e-4)
        self.assertAlmostEqual(max(lrs), 1e-4)
        self.assertTrue(all(a >= b for a, b in zip(lrs[100:], lrs[101:])))
        self.assertAlmostEqual(train.finetune_lr(1000, 1e-4, 100, 1000), 1e-5)

    def test_train_only_postnet_keeps_the_rest_in_eval_mode(self):
        net = tiny_decoder()
        train.train_mode(net, "postnet")
        self.assertTrue(net.postnet.training)
        self.assertFalse(net.encoder.training or net.decoder.training or net.variance_adaptor.training)
        train.train_mode(net)
        self.assertTrue(net.encoder.training and net.postnet.training)


class BuzzMeasures(unittest.TestCase):
    SR = 22050

    def voice(self, seconds=1.0):
        t = np.arange(int(seconds * self.SR)) / self.SR
        return (0.1 * sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 50))).astype(np.float32)

    def test_flatness_is_low_for_harmonics_and_high_for_noise(self):
        voice = train.band_flatness(self.voice(), self.SR)
        noise = train.band_flatness(np.random.default_rng(0).standard_normal(self.SR).astype(np.float32), self.SR)
        self.assertEqual(len(voice), 3)
        self.assertTrue(all(v < 0.1 for v in voice), voice)
        self.assertTrue(all(n > 0.4 for n in noise), noise)

    def test_flatness_rises_with_buzz(self):
        x = self.voice()
        noise = np.random.default_rng(0).standard_normal(len(x)).astype(np.float32) * 0.003
        clean, buzzy = train.band_flatness(x, self.SR), train.band_flatness(x + noise, self.SR)
        self.assertTrue(all(b > c for b, c in zip(buzzy, clean)), (clean, buzzy))

    def test_flatness_uses_the_loudest_half_of_the_frames(self):
        x = self.voice()
        quiet = np.random.default_rng(0).standard_normal(len(x) // 2).astype(np.float32) * 1e-4
        np.testing.assert_allclose(train.band_flatness(np.concatenate([x, quiet]), self.SR),
                                   train.band_flatness(x, self.SR), atol=0.02)

    def test_sharpness_is_lower_for_a_blurred_mel(self):
        rng = np.random.default_rng(0)
        mel = rng.standard_normal((50, 80))
        blurred = (mel[:, :-2] + mel[:, 1:-1] + mel[:, 2:]) / 3
        self.assertGreater(train.mel_sharpness(mel), 2 * train.mel_sharpness(blurred))
        self.assertEqual(train.mel_sharpness(np.ones((5, 80))), 0.0)


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

    def test_fine_tunes_the_postnet_of_a_finished_run(self):
        init = os.path.join(self.tmp.name, "finished.pt")  # a copy: test_resumes changes the run
        shutil.copy(os.path.join(self.run_dir, "decoder_best.pt"), init)
        run = os.path.join(self.tmp.name, "postnet")
        result = subprocess.run(
            [sys.executable, os.path.join(REPO, "decoder", "train.py"), self.data, run, "--init", init,
             "--reset-postnet", "--train-only", "postnet", "--postnet-final-dropout", "0", "--lr", "1e-4", "--warmup", "1",
             "--keep-checkpoints", "--steps", "2", "--batch-size", "2", "--eval-every", "1", "--save-every", "2",
             "--samples", "1", "--workers", "0", "--device", "cpu"], capture_output=True, text=True, timeout=900)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        self.assertIn("reset the postnet (final-norm)", result.stdout)
        self.assertIn("training only the postnet", result.stdout)
        self.assertRegex(result.stdout, r"buzz \(pitch predicted\): flatness 0.5-2/2-4/4-8 kHz [0-9.]+/[0-9.]+/[0-9.]+ \(vocoded target [0-9.]+/")
        before = torch.load(init, map_location="cpu")["model"]
        for step in (1, 2):
            after = torch.load(os.path.join(run, "checkpoints", f"step_{step:06d}.pt"), map_location="cpu")
            self.assertEqual(after["step"], step)
            for k, v in after["model"].items():
                if not k.startswith("postnet."):
                    self.assertTrue(torch.equal(v, before[k]), k)
        self.assertFalse(torch.equal(after["model"]["postnet.convolutions.0.0.conv.weight"],
                                     before["postnet.convolutions.0.0.conv.weight"]))
        with open(os.path.join(run, "history.json")) as f:
            val = json.load(f)["val"]
        self.assertEqual([v["step"] for v in val], [1, 2])
        for key in ("flatness", "sharpness", "sharpness_real", "mel_inference"):
            self.assertIn(key, val[-1])

    def test_benchmark_prints_the_speed_and_saves_nothing(self):
        run = os.path.join(self.tmp.name, "benchmark")
        result = subprocess.run(
            [sys.executable, os.path.join(REPO, "decoder", "train.py"), self.data, run, "--benchmark", "5",
             "--steps", "3000", "--batch-size", "2", "--workers", "0", "--device", "cpu"],
            capture_output=True, text=True, timeout=900)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        self.assertRegex(result.stdout, r"benchmark: [0-9.]+ s/step .* over 2 steps at batch 2; 3000 steps .* would take")
        self.assertFalse(any(os.path.exists(os.path.join(run, n)) for n in ("latest.pt", "decoder_best.pt", "history.json")))

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



BIGVGAN_CACHE = os.path.join(CACHE, "bigvgan", "bigvgan_v2_22khz_80band_fmax8k_256x", "bigvgan_generator.pt")


@unittest.skipUnless(os.environ.get("WESPER_DECODER_SMOKE") and os.environ.get("WESPER_BIGVGAN"),
                     "set WESPER_DECODER_SMOKE=1 and WESPER_BIGVGAN=1 to train a BigVGAN decoder on fake data")
class BigVGANSmoke(unittest.TestCase):
    """The whole decoder pipeline for --vocoder bigvgan22k: prepare, train, convert with WESPER."""

    @classmethod
    def setUpClass(cls):
        if not all(os.path.exists(p) for p in NEEDED + [BIGVGAN_CACHE]):
            raise unittest.SkipTest("WESPER's and BigVGAN's checkpoints aren't cached yet")
        from test_prepare_data import make_recordings
        cls.tmp = tempfile.TemporaryDirectory()
        cls.book, cls.data, cls.run_dir = (os.path.join(cls.tmp.name, d) for d in ("book", "data", "run"))
        make_recordings(cls.book)
        cls.prepared = cls.prepare("bigvgan22k")
        cls.trained = subprocess.run(
            [sys.executable, os.path.join(REPO, "decoder", "train.py"), cls.data, cls.run_dir, "--steps", "2",
             "--batch-size", "2", "--eval-every", "2", "--save-every", "2", "--samples", "1", "--workers", "0",
             "--device", "cpu"], capture_output=True, text=True, timeout=900)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def prepare(cls, vocoder):
        return subprocess.run([sys.executable, os.path.join(REPO, "decoder", "prepare_data.py"), cls.book, cls.data,
                               "--val-recordings", "1", "--vocoder", vocoder], capture_output=True, text=True, timeout=600)

    def test_prepares_mels_at_bigvgans_frame_rate(self):
        import csv
        import json
        import vocoders
        self.assertEqual(self.prepared.returncode, 0, self.prepared.stderr[-2000:])
        with open(os.path.join(self.data, "prep.json")) as f:
            self.assertEqual(json.load(f)["vocoder"], "bigvgan22k")
        s = vocoders.spec("bigvgan22k")
        with open(os.path.join(self.data, "segments.tsv"), newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        for r in rows:
            with np.load(os.path.join(self.data, "segments", r["id"] + ".npz")) as d:
                n = int(r["frames"])
                self.assertEqual(d["mel"].shape, (80, n))
                self.assertEqual(d["pitch"].shape, (n,))
                self.assertLessEqual(n, vocoders.n_frames(len(d["units"]), s))
                self.assertGreaterEqual(n, vocoders.n_frames(len(d["units"]), s) - 2)
                self.assertLessEqual(n, 1000)  # FastSpeech2's max_seq_len

    def test_refuses_to_mix_vocoders_in_one_folder(self):
        result = self.prepare("hifigan16k")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("prepared for bigvgan22k", result.stderr)

    def test_trains_and_writes_22khz_samples(self):
        import soundfile as sf
        self.assertEqual(self.trained.returncode, 0, self.trained.stderr[-3000:])
        self.assertIn("vocoder: bigvgan22k", self.trained.stdout)
        for folder in ("vocoded-target", "step_000002"):
            (name,) = os.listdir(os.path.join(self.run_dir, "samples", folder))
            self.assertEqual(sf.info(os.path.join(self.run_dir, "samples", folder, name)).samplerate, 22050, folder)

    def test_wesper_converts_with_bigvgan(self):
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
        self.assertEqual((w2n.vocoder_spec.name, w2n.sample_rate), ("bigvgan22k", 22050))
        seconds = len(out) / w2n.sample_rate
        self.assertAlmostEqual(seconds, len(x) / 16000, delta=0.25)  # durations stay 1, at the new frame rate


if __name__ == "__main__":
    unittest.main()
