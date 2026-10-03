"""Tests for web/export_models.py, which exports WESPER's models to ONNX for the browser demo.

The wrapper tests check each rewritten operation against the PyTorch code it replaces, and that
small exported graphs keep their lengths dynamic. The model tests need WESPER's checkpoints in
torch's hub cache (any earlier conversion run puts them there) and are skipped otherwise. The
export smoke test runs the whole script (~2 min) and is opt-in:

    .venv/bin/python -m unittest tests.test_export_models -v
    WESPER_EXPORT_SMOKE=1 .venv/bin/python -m unittest tests.test_export_models -v

A watchdog dumps stacks and exits if the module runs longer than WESPER_EXPORT_TEST_TIMEOUT
seconds (default 900).
"""
import faulthandler
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(torch.hub.get_dir(), "checkpoints")
CACHED = [os.path.join(CACHE, f) for f in ("model-layer12-450000.pt", "googletts_neutral_best.tar", "g_00205000")]
SV_ENCODER = os.path.join(REPO, "colab", "data", "runs", "n2w-finetune", "encoder_best.pt")

sys.path.insert(0, os.path.join(REPO, "web"))
import export_models as em  # noqa: E402
from model.modules import LengthRegulator  # noqa: E402  (FastSpeech2, on the path via export_models)

_cwd = os.getcwd()


def setUpModule():
    faulthandler.dump_traceback_later(int(os.environ.get("WESPER_EXPORT_TEST_TIMEOUT", 900)), exit=True)
    os.chdir(REPO)  # FastSpeech2 and HiFi-GAN read their configs by paths relative to the repo
    torch.set_grad_enabled(False)


def tearDownModule():
    os.chdir(_cwd)
    faulthandler.cancel_dump_traceback_later()


def run_onnx(module, example, inputs, input_name="x"):
    """Export module traced on `example`, then run the ONNX graph on each of `inputs`."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.onnx")
        torch.onnx.export(module, (example,), path, opset_version=em.OPSET, input_names=[input_name],
                          output_names=["y"], dynamic_axes={input_name: {1: "n"}, "y": {1: "m"}})
        sess = em.session(path)
        return [sess.run(None, {input_name: x.numpy()})[0] for x in inputs]


class Wrappers(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_encoder_layer_matches_torch(self):
        layer = nn.TransformerEncoderLayer(768, 12, 3072, activation="gelu", batch_first=True).eval()
        wrapped = em.ExportableEncoderLayer(layer, 12)
        for n in (1, 37, 150):
            x = torch.randn(1, n, 768)
            torch.testing.assert_close(wrapped(x), layer(x), atol=1e-5, rtol=1e-4)

    def test_encoder_layer_export_keeps_length_dynamic(self):
        layer = nn.TransformerEncoderLayer(16, 2, 32, activation="gelu", batch_first=True).eval()
        wrapped = em.ExportableEncoderLayer(layer, 2).eval()
        inputs = [torch.randn(1, n, 16) for n in (7, 23, 3)]
        for x, y in zip(inputs, run_onnx(wrapped, torch.randn(1, 11, 16), inputs)):
            np.testing.assert_allclose(y, layer(x).numpy(), atol=1e-5)

    def test_bucketize_matches_torch(self):
        bins = torch.linspace(-2.0, 3.0, 255)
        x = torch.cat([bins, bins + 1e-4, bins - 1e-4, torch.tensor([-10.0, 10.0]), torch.randn(500)])[None]
        torch.testing.assert_close(em.bucketize(x, bins), torch.bucketize(x, bins), rtol=0, atol=0)

    def test_regulate_length_matches_fastspeech2(self):
        x = torch.randn(1, 6, 4)
        for d in ([1, 1, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1], [2, 0, 0, 0, 0, 5]):
            d = torch.tensor(d)
            expected, mel_len = LengthRegulator().LR(x, d[None], None)
            torch.testing.assert_close(em.regulate_length(x, d), expected)
            self.assertEqual(mel_len.item(), d.sum().item())

    def test_regulate_length_export_follows_the_input(self):
        class M(nn.Module):  # durations computed from the input, as in the decoder
            def forward(self, x):
                return em.regulate_length(x, torch.clamp(torch.round(x[0, :, 0] * 2), min=0).long())
        m = M()
        inputs = [torch.tensor([[[0.6], [-1.0], [1.4], [0.4]]]), torch.tensor([[[1.0], [0.1], [0.3], [2.2], [0.5], [0.0]]])]
        for x, y in zip(inputs, run_onnx(m, torch.rand(1, 5, 1), inputs)):
            np.testing.assert_array_equal(y, m(x).numpy())


@unittest.skipUnless(all(os.path.exists(p) for p in CACHED), "WESPER's checkpoints aren't cached yet")
class AgainstCheckpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hubert = em.wn.load_hubert(CACHED[0], device="cpu")
        cls.configs = configs = em.load_configs()
        cls.fs2 = em.wn.load_fastspeech2(configs, checkpoint_path=CACHED[1], device="cpu")
        cls.vocoder = em.wn.load_hifigan(configs[1], checkpoint_path=CACHED[2], device="cpu")
        cls.wav, _ = sf.read(os.path.join(REPO, "sample_whisper.wav"), dtype="float32")
        cls.units = em.reference_units(cls.hubert, cls.wav)

    def test_encoder_wrapper_matches_wav2units(self):
        units = em.EncoderExport(self.hubert).eval()(torch.from_numpy(self.wav)[None])
        self.assertEqual(units.shape, (1, len(self.wav) // em.HOP, 256))
        torch.testing.assert_close(units, self.units, atol=1e-4, rtol=1e-4)

    def test_encoder_wrapper_leaves_the_model_unchanged(self):
        em.EncoderExport(self.hubert)
        torch.testing.assert_close(em.reference_units(self.hubert, self.wav), self.units, rtol=0, atol=0)

    def test_reference_wav_is_units2wav(self):
        """The export checks compare against reference_wav(); it must be what the GUI plays."""
        gui, _ = em.wn.units2wav(self.units, self.fs2, self.vocoder, self.configs[1], self.configs[0], device="cpu")
        ref = (em.reference_wav(self.fs2, self.vocoder, self.units).numpy() * 32768.0).astype("int16")
        np.testing.assert_array_equal(ref, gui)

    def test_decoder_wrapper_matches_reference(self):
        wav = em.DecoderExport(self.fs2, self.vocoder).eval()(self.units)[0]
        torch.testing.assert_close(wav, em.reference_wav(self.fs2, self.vocoder, self.units), atol=1e-5, rtol=0)

    def test_decoder_wrapper_beyond_the_original_position_table(self):
        units = self.units.repeat(1, 12, 1)  # 1104 frames: past FastSpeech2's 1000-frame table
        self.assertGreater(units.shape[1], 1000)
        wav = em.DecoderExport(self.fs2, self.vocoder).eval()(units)[0]
        torch.testing.assert_close(wav, em.reference_wav(self.fs2, self.vocoder, units), atol=1e-5, rtol=0)


@unittest.skipUnless(os.environ.get("WESPER_EXPORT_SMOKE"), "set WESPER_EXPORT_SMOKE=1 to run the full export")
@unittest.skipUnless(all(os.path.exists(p) for p in CACHED), "WESPER's checkpoints aren't cached yet")
class ExportSmoke(unittest.TestCase):
    def test_full_export(self):
        has_sv = os.path.exists(SV_ENCODER)
        with tempfile.TemporaryDirectory() as out:
            args = [sys.executable, os.path.join(REPO, "web", "export_models.py"), "--out", out, "--timeout", "800"]
            args += ["--original", CACHED[0], "--fastspeech2", CACHED[1], "--hifigan", CACHED[2]]
            if not has_sv:
                args += ["--sv", ""]
            r = subprocess.run(args, capture_output=True, text=True, timeout=900, cwd=tempfile.gettempdir())
            self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])

            manifest = json.load(open(os.path.join(out, "models.json")))
            self.assertEqual([e["id"] for e in manifest["encoders"]], ["sv", "original"] if has_sv else ["original"])
            self.assertEqual((manifest["sampleRate"], manifest["hop"]), (16000, 320))
            original = manifest["encoders"][-1]
            self.assertEqual((original["targetDbfs"], original["maxGainDb"]), (None, None))
            if has_sv:
                self.assertEqual((manifest["encoders"][0]["targetDbfs"], manifest["encoders"][0]["maxGainDb"]), (-20.0, 40.0))
            for entry in manifest["encoders"] + manifest["decoders"]:
                for precision in ("fp32", "fp16"):
                    info = entry["files"][precision]
                    path = os.path.join(out, info["path"])
                    self.assertEqual(os.path.getsize(path), info["bytes"])
                    self.assertEqual(hashlib.sha256(open(path, "rb").read()).hexdigest(), info["sha256"])
                    self.assertTrue(entry["checks"][precision])


if __name__ == "__main__":
    unittest.main()
