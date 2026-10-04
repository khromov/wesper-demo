"""Tests for decoder/finetune_vocoder.py.

The pipeline test fine-tunes a tiny BigVGAN for a few steps on CPU: fake recordings prepared for
BigVGAN's settings, exported at 22.05 kHz, and a decoder run with random weights. Nothing is
downloaded (--init none, --config with a tiny model).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch
import yaml

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(TESTS))
sys.path.insert(0, REPO)
sys.path.insert(0, TESTS)

import vocoders  # noqa: E402
from decoder import export_vocoder_audio as eva, finetune_vocoder as ft, train as decoder_train  # noqa: E402
from test_export_vocoder_audio import BIG, prepare  # noqa: E402
from test_prepare_data import make_recordings  # noqa: E402

SCRIPT = os.path.join(REPO, "decoder", "finetune_vocoder.py")
# bigvgan22k's audio, mel and training settings, with a tiny generator and discriminators.
TINY = {
    "resblock": "1", "upsample_rates": [4, 4, 4, 4], "upsample_kernel_sizes": [8, 8, 8, 8], "upsample_initial_channel": 32,
    "resblock_kernel_sizes": [3], "resblock_dilation_sizes": [[1, 3]], "use_tanh_at_final": False,
    "use_bias_at_final": False, "activation": "snakebeta", "snake_logscale": True,
    "use_cqtd_instead_of_mrd": True, "cqtd_filters": 8, "cqtd_max_filters": 16, "cqtd_filters_scale": 1,
    "cqtd_dilations": [1, 2], "cqtd_hop_lengths": [256], "cqtd_n_octaves": [6], "cqtd_bins_per_octaves": [12],
    "mpd_reshapes": [2, 3], "use_spectral_norm": False, "discriminator_channel_mult": 1,
    "use_multiscale_melloss": True, "lambda_melloss": 15, "clip_grad_norm": 500, "adam_b1": 0.8, "adam_b2": 0.99,
    "num_mels": 80, "n_fft": 1024, "hop_size": 256, "win_size": 1024, "sampling_rate": 22050, "fmin": 0, "fmax": 8000,
}


def decoder_run(run_dir, data_dir, voc):
    """A decoder/train.py run folder with a randomly initialized decoder."""
    import model.modules
    import utils.tools
    from model import FastSpeech2
    utils.tools.device = model.modules.device = "cpu"
    os.makedirs(run_dir)
    shutil.copy(os.path.join(data_dir, "stats.json"), run_dir)
    pre, model_config, _ = decoder_train.configs(run_dir, voc)
    with open(os.path.join(run_dir, "preprocess.yaml"), "w") as f:
        yaml.safe_dump(pre, f, sort_keys=False)
    pre["path"]["preprocessed_path"] = run_dir
    torch.manual_seed(0)
    torch.save({"model": FastSpeech2(pre, model_config).state_dict()}, os.path.join(run_dir, "decoder_best.pt"))


def finetune(data, audio, run, out, *extra):
    return subprocess.run([sys.executable, SCRIPT, data, audio, run, out, "--init", "none", "--config", CONFIG[0],
                           "--batch-size", "2", "--segment-frames", "32", "--eval-every", "2", "--save-every", "2",
                           "--workers", "0", "--device", "cpu", *extra],
                          capture_output=True, text=True, timeout=900)


CONFIG = []  # path of TINY as a file, set in setUpModule


def setUpModule():
    global TMP
    TMP = tempfile.TemporaryDirectory()
    CONFIG.append(os.path.join(TMP.name, "tiny.json"))
    with open(CONFIG[0], "w") as f:
        json.dump(TINY, f)


def tearDownModule():
    TMP.cleanup()


class Crops(unittest.TestCase):
    def test_mel_frames_and_audio_line_up(self):
        # Frame t of the mel is t, and samples t * hop to (t + 1) * hop of the audio are t / 1000.
        with tempfile.TemporaryDirectory() as tmp:
            frames, hop = 100, 256
            np.save(os.path.join(tmp, "a.npy"), np.tile(np.arange(frames, dtype=np.float16), (80, 1)))
            wav = np.repeat(np.arange(frames) / 1000, hop) * 10 ** (-eva.HEADROOM_DB / 20)
            sf.write(os.path.join(tmp, "a.flac"), wav, 22050, subtype="PCM_24")
            crops = ft.Crops(["a"], tmp, tmp, 32, hop)
            for _ in range(20):
                mel, audio = crops[0]
                self.assertEqual((tuple(mel.shape), tuple(audio.shape)), ((80, 32), (1, 32 * hop)))
                np.testing.assert_allclose(audio[0].numpy().reshape(32, hop).mean(1) * 1000, mel[0].numpy(), atol=1e-3)


class FlatnessGap(unittest.TestCase):
    SR = 22050

    def voice(self, seconds=1.0):
        t = np.arange(int(seconds * self.SR)) / self.SR
        return (0.1 * sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 50))).astype(np.float32)

    def test_zero_for_the_recording_itself(self):
        x = self.voice()
        self.assertAlmostEqual(ft.flatness_gap(x, x, self.SR), 0.0)

    def test_grows_with_buzz(self):
        # Noise on the harmonics makes speech flatter: the more of it, the bigger the gap.
        x = self.voice()
        noise = np.random.default_rng(0).standard_normal(len(x)).astype(np.float32) * 0.01
        a, b = ft.flatness_gap(x + noise, x, self.SR), ft.flatness_gap(x + 4 * noise, x, self.SR)
        self.assertGreater(a, 0.01)
        self.assertGreater(b, a)

    def test_ignores_level(self):
        x = self.voice()
        self.assertAlmostEqual(ft.flatness_gap(0.5 * x, x, self.SR), 0.0, places=4)


class Resample(unittest.TestCase):
    """libs/bigvgan/resample.py, the port of torchaudio's Resample that the CQT discriminator uses."""

    def test_matches_torchaudio(self):
        try:
            import torchaudio
        except ImportError:
            self.skipTest("torchaudio isn't installed")
        from libs.bigvgan.resample import Resample
        x = torch.randn(3, 1, 5003)
        for orig, new in ((22050, 44100), (24000, 48000), (16000, 22050)):
            ours, theirs = Resample(orig, new), torchaudio.transforms.Resample(orig, new)
            torch.testing.assert_close(ours.kernel, theirs.kernel, rtol=0, atol=0)
            torch.testing.assert_close(ours(x), theirs(x), rtol=0, atol=0)

    def test_doubles_the_rate_of_a_tone(self):
        from libs.bigvgan.resample import Resample
        sr = 22050
        t = torch.arange(sr) / sr
        y = Resample(sr, 2 * sr)(torch.sin(2 * np.pi * 440 * t)[None])[0]
        self.assertEqual(len(y), 2 * sr)
        expected = torch.sin(2 * np.pi * 440 * torch.arange(2 * sr) / (2 * sr))
        torch.testing.assert_close(y[1000:-1000], expected[1000:-1000], atol=2e-3, rtol=0)

    def test_the_cqt_discriminator_needs_no_torchaudio(self):
        code = ("import sys; sys.modules['torchaudio'] = None\n"
                "from libs.bigvgan.discriminators import MultiScaleSubbandCQTDiscriminator\n"
                f"from libs.bigvgan.env import AttrDict; import json; MultiScaleSubbandCQTDiscriminator(AttrDict(json.load(open({CONFIG[0]!r}))))")
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])


class LoadDiscriminator(unittest.TestCase):
    def test_tolerates_checkpoints_with_or_without_resampling_kernels(self):
        from libs.bigvgan.env import AttrDict
        _, mrd = ft.discriminators(AttrDict(TINY))
        state = mrd.state_dict()
        kernels = [k for k in state if k.endswith("resample.kernel")]
        self.assertTrue(kernels)
        ft.load_discriminator(mrd, state)
        ft.load_discriminator(mrd, {k: v for k, v in state.items() if k not in kernels})
        with self.assertRaises(AssertionError):
            ft.load_discriminator(mrd, {k: v for k, v in state.items() if "conv_post" not in k})


class RocmWrapper(unittest.TestCase):
    def test_runs_the_script_with_miopen_off_and_its_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "s.py")
            with open(script, "w") as f:
                f.write("import sys, torch\nprint(torch.backends.cudnn.enabled, sys.argv[1:], __name__)\n")
            r = subprocess.run([sys.executable, os.path.join(REPO, "decoder", "rocm.py"), script, "a", "--b", "c"],
                               capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "False ['a', '--b', 'c'] __main__")

    def test_explains_itself_without_a_script(self):
        r = subprocess.run([sys.executable, os.path.join(REPO, "decoder", "rocm.py")], capture_output=True, text=True, timeout=120)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("MIOpen", r.stderr)


class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        src = os.path.join(cls.tmp.name, "book")
        make_recordings(src, n=2)
        cls.data = os.path.join(cls.tmp.name, "data")
        cls.rows = prepare(src, cls.data, BIG, val={"chapter_001"})
        cls.audio = os.path.join(cls.tmp.name, "audio22k")
        eva.main([src, cls.data, cls.audio])
        cls.run_dir = os.path.join(cls.tmp.name, "run")
        decoder_run(cls.run_dir, cls.data, BIG)
        cls.out = os.path.join(cls.tmp.name, "run-vft")
        cls.result = finetune(cls.data, cls.audio, cls.run_dir, cls.out, "--steps", "4")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_succeeds(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr[-3000:])
        self.assertIn(f"decoder mels: {len(self.rows)} computed", self.result.stdout)
        self.assertIn("val flatness gap", self.result.stdout)
        self.assertIn("done at step 4", self.result.stdout)

    def test_caches_the_decoders_mel_of_each_utterance(self):
        for r in self.rows:
            mel = np.load(os.path.join(self.out, "mels", r["id"] + ".npy"))
            self.assertEqual((mel.shape, mel.dtype), ((80, int(r["frames"])), np.float16))

    def test_cached_mels_are_the_decoder_with_the_recordings_pitch(self):
        # The cache holds the decoder's output given the recording's own pitch and energy, not predicted ones.
        net, _ = ft.load_decoder(self.run_dir, "cpu")
        with open(os.path.join(self.run_dir, "stats.json")) as f:
            stats = json.load(f)
        dataset = decoder_train.Utterances(self.data, self.rows, stats, BIG)
        batch = decoder_train.collate([dataset[0]])
        with torch.no_grad():
            expected = net(*batch[2:])[1][0].numpy().T
        cached = np.load(os.path.join(self.out, "mels", self.rows[0]["id"] + ".npy")).astype(np.float32)
        np.testing.assert_allclose(cached, expected, atol=0.02)

    def test_output_is_a_decoder_run_with_its_own_vocoder(self):
        for name in ("decoder_best.pt", "stats.json", "bigvgan_generator.pt", "latest.pt", "history.json"):
            self.assertTrue(os.path.exists(os.path.join(self.out, name)), name)
        with open(os.path.join(self.out, "preprocess.yaml")) as f:
            pre = yaml.safe_load(f)
        self.assertEqual(pre["vocoder"], {"name": "bigvgan22k", "checkpoint": "bigvgan_generator.pt"})
        self.assertEqual(pre["path"]["preprocessed_path"], os.path.relpath(os.path.realpath(self.out), REPO))
        self.assertEqual(vocoders.run_checkpoint(pre, self.out), os.path.join(self.out, "bigvgan_generator.pt"))
        from libs.bigvgan.bigvgan import BigVGAN
        from libs.bigvgan.env import AttrDict
        state = torch.load(os.path.join(self.out, "bigvgan_generator.pt"), map_location="cpu")
        BigVGAN(AttrDict(TINY), use_cuda_kernel=False).load_state_dict(state["generator"], strict=True)

    def test_validates_from_the_original_vocoder_on(self):
        with open(os.path.join(self.out, "history.json")) as f:
            history = json.load(f)
        self.assertEqual([v["step"] for v in history["val"]], [0, 2, 4])
        self.assertEqual([t["step"] for t in history["train"]], [4])
        self.assertTrue(all(np.isfinite(v["mel"]) and np.isfinite(v["flatness_gap"]) for v in history["val"]))

    def test_writes_samples_for_each_validation(self):
        val = [r["id"] for r in self.rows if r["split"] == "val"][:ft.SAMPLES]
        for folder in ("reference", "step_000000", "step_000002", "step_000004"):
            path = os.path.join(self.out, "samples", folder)
            self.assertEqual(sorted(os.listdir(path)), sorted(v + ".wav" for v in val), folder)
            self.assertEqual(sf.info(os.path.join(path, val[0] + ".wav")).samplerate, 22050)

    def test_resumes(self):
        out = os.path.join(self.tmp.name, "run-vft-resumed")  # a copy, so the other tests see the 4-step run
        shutil.copytree(self.out, out)
        result = finetune(self.data, self.audio, self.run_dir, out, "--steps", "6")
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        self.assertIn("decoder mels: 0 computed", result.stdout)
        self.assertIn("resuming from step 4", result.stdout)
        self.assertIn("done at step 6", result.stdout)

    def test_another_decoder_checkpoint(self):
        # --decoder-checkpoint: e.g. one of train.py --keep-checkpoints' checkpoints/step_NNNNNN.pt
        checkpoint = os.path.join(self.tmp.name, "step_000002.pt")
        state = torch.load(os.path.join(self.run_dir, "decoder_best.pt"), map_location="cpu")
        state["model"] = {k: v * 0.5 if v.is_floating_point() else v for k, v in state["model"].items()}
        torch.save({**state, "step": 2}, checkpoint)
        out = os.path.join(self.tmp.name, "run-vft-step2")
        result = finetune(self.data, self.audio, self.run_dir, out, "--steps", "2", "--decoder-checkpoint", checkpoint)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        copied = torch.load(os.path.join(out, "decoder_best.pt"), map_location="cpu")
        self.assertEqual(copied["step"], 2)
        r = self.rows[0]["id"]
        self.assertFalse(np.array_equal(np.load(os.path.join(out, "mels", r + ".npy")),
                                        np.load(os.path.join(self.out, "mels", r + ".npy"))))

    def test_benchmark_prints_the_speed_and_saves_nothing(self):
        out = os.path.join(self.tmp.name, "run-vft-benchmark")
        result = finetune(self.data, self.audio, self.run_dir, out, "--benchmark", "5", "--steps", "1000")
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        self.assertRegex(result.stdout, r"benchmark: [0-9.]+ s/step .* over 2 steps .* 1000 steps \(--steps 1000\) would take")
        self.assertFalse(any(os.path.exists(os.path.join(out, n)) for n in ("latest.pt", "bigvgan_generator.pt", "history.json")))
        self.assertFalse(os.path.exists(os.path.join(out, "samples", "reference")))

    def test_refuses_hifigan(self):
        run = os.path.join(self.tmp.name, "hifigan-run")
        os.makedirs(run)
        for name in ("decoder_best.pt", "stats.json"):
            shutil.copy(os.path.join(self.run_dir, name), run)
        pre, _, _ = decoder_train.configs(run, vocoders.HIFIGAN16K)
        with open(os.path.join(run, "preprocess.yaml"), "w") as f:
            yaml.safe_dump(pre, f)
        result = finetune(self.data, self.audio, run, os.path.join(self.tmp.name, "hifigan-vft"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("only BigVGAN can be fine-tuned", result.stderr)


if __name__ == "__main__":
    unittest.main()
