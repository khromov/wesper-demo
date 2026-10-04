"""Tests for web/export_models.py, which exports WESPER's models to ONNX for the browser demo.

The wrapper tests check each rewritten operation against the PyTorch code it replaces, and that
small exported graphs keep their lengths dynamic. The model tests need WESPER's checkpoints in
torch's hub cache (any earlier conversion run puts them there) and are skipped otherwise. The
BigVGAN tests run its 449 MB vocoder and are opt-in, like those in test_vocoders.py; without a
BigVGAN-trained run in decoder/runs, they use a stand-in: the narrator's HiFi-GAN-trained weights
set up for BigVGAN (stand_in_bigvgan_run), which runs the same graph and sounds rougher. The
export smoke test runs the whole script (~2 min, ~5 with WESPER_BIGVGAN) and is opt-in too:

    .venv/bin/python -m unittest tests.test_export_models -v
    WESPER_BIGVGAN=1 .venv/bin/python -m unittest tests.test_export_models -v
    WESPER_EXPORT_SMOKE=1 WESPER_BIGVGAN=1 .venv/bin/python -m unittest tests.test_export_models -v

A watchdog dumps stacks and exits if the module runs longer than WESPER_EXPORT_TEST_TIMEOUT
seconds (default 1800).
"""
import dataclasses
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
SV_DECODER = os.path.join(REPO, "decoder", "runs", "sv-narrator")  # decoder/train.py run folder
SV_BIGVGAN_DECODER = os.path.join(REPO, "decoder", "runs", "sv-narrator-bigvgan22k")
BIGVGAN = os.path.join(CACHE, "bigvgan", "bigvgan_v2_22khz_80band_fmax8k_256x")
WITH_BIGVGAN = bool(os.environ.get("WESPER_BIGVGAN")) and os.path.exists(os.path.join(BIGVGAN, "bigvgan_generator.pt"))

sys.path.insert(0, os.path.join(REPO, "web"))
import export_models as em  # noqa: E402
from model.modules import LengthRegulator  # noqa: E402  (FastSpeech2, on the path via export_models)
from decoder import train  # noqa: E402  (the repo is on the path via export_models)

# bigvgan22k's audio and frame settings, without fetching its config.json
BIGVGAN22K = em.vocoders.Spec("bigvgan22k", 22050, 1024, 256, 1024, 80, 0, 8000, "")


def stand_in_bigvgan_run(run):
    """A decoder/train.py run folder for BigVGAN made from the HiFi-GAN narrator run: its weights,
    with the preprocess.yaml train.py writes for bigvgan22k. For testing the export only."""
    os.makedirs(run, exist_ok=True)
    for f in ("decoder_best.pt", "stats.json"):
        os.symlink(os.path.join(SV_DECODER, f), os.path.join(run, f))
    with open(os.path.join(run, "preprocess.yaml"), "w") as f:
        em.yaml.safe_dump(train.configs(run, em.vocoders.spec("bigvgan22k"))[0], f, sort_keys=False)
    return run

_cwd = os.getcwd()


def setUpModule():
    faulthandler.dump_traceback_later(int(os.environ.get("WESPER_EXPORT_TEST_TIMEOUT", 1800)), exit=True)
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

    def test_units_to_frames_matches_vocoders(self):
        for n in (1, 2, 7, 92, 2304, 6000):  # 2304: n_frames is an exact integer there; 6000: 2 minutes
            units = torch.randn(n, 8)
            expected = em.vocoders.units_to_frames(units.numpy(), em.vocoders.n_frames(n, BIGVGAN22K), BIGVGAN22K)
            np.testing.assert_allclose(em.units_to_frames(units[None], BIGVGAN22K)[0].numpy(), expected, atol=1e-6)

    def test_units_to_frames_export_follows_the_input(self):
        class M(nn.Module):
            def forward(self, x):
                return em.units_to_frames(x, BIGVGAN22K)
        inputs = [torch.randn(1, n, 8) for n in (1, 7, 92, 301)]
        for x, y in zip(inputs, run_onnx(M(), torch.randn(1, 50, 8), inputs)):
            self.assertEqual(y.shape[1], em.vocoders.n_frames(x.shape[1], BIGVGAN22K))
            np.testing.assert_allclose(y, M()(x).numpy(), atol=1e-6)


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


class RunConfigs(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(os.path.join(BIGVGAN, "config.json")), "BigVGAN's config.json isn't cached yet")
    def test_reads_a_bigvgan_runs_vocoder(self):
        with tempfile.TemporaryDirectory() as run:
            with open(os.path.join(run, "preprocess.yaml"), "w") as f:
                f.write("path: {preprocessed_path: x}\nvocoder: {name: bigvgan22k}\n")
            voc = em.run_vocoder(em.load_run_configs(run, {})[0])
            self.assertEqual((voc.name, voc.sample_rate, voc.hop, voc.n_mels), ("bigvgan22k", 22050, 256, 80))
            self.assertEqual(voc, dataclasses.replace(BIGVGAN22K, checkpoint=voc.checkpoint))

    def test_runs_without_a_vocoder_are_hifigan(self):
        with tempfile.TemporaryDirectory() as run:
            with open(os.path.join(run, "preprocess.yaml"), "w") as f:
                f.write("path: {preprocessed_path: x}\n")
            self.assertEqual(em.run_vocoder(em.load_run_configs(run, {})[0]), em.vocoders.HIFIGAN16K)

    def test_finds_stats_in_the_run_folder_wherever_it_is(self):
        with tempfile.TemporaryDirectory() as run:
            with open(os.path.join(run, "preprocess.yaml"), "w") as f:
                f.write("path: {preprocessed_path: decoder/runs/elsewhere}\n")
            pre, model = em.load_run_configs(run, {"m": 1})
            self.assertEqual((pre["path"]["preprocessed_path"], model), (run, {"m": 1}))


@unittest.skipUnless(all(os.path.exists(p) for p in CACHED), "WESPER's checkpoints aren't cached yet")
@unittest.skipUnless(os.path.exists(os.path.join(SV_DECODER, "decoder_best.pt")), "no Swedish decoder run in decoder/runs/sv-narrator")
class NarratorDecoder(unittest.TestCase):
    """The Swedish narrator decoder from decoder/train.py, exported the same way as googletts."""

    @classmethod
    def setUpClass(cls):
        hubert = em.wn.load_hubert(CACHED[0], device="cpu")
        configs = em.load_run_configs(os.path.abspath(SV_DECODER), em.load_configs()[1])
        cls.fs2 = em.wn.load_fastspeech2(configs, checkpoint_path=os.path.join(SV_DECODER, "decoder_best.pt"), device="cpu")
        cls.vocoder = em.wn.load_hifigan(configs[1], checkpoint_path=CACHED[2], device="cpu")
        wav, _ = sf.read(os.path.join(REPO, "sample_whisper.wav"), dtype="float32")
        cls.units = em.reference_units(hubert, wav)

    def test_decoder_wrapper_matches_reference(self):
        wav = em.DecoderExport(self.fs2, self.vocoder).eval()(self.units)[0]
        torch.testing.assert_close(wav, em.reference_wav(self.fs2, self.vocoder, self.units), atol=1e-5, rtol=0)

    def test_one_frame_per_unit(self):
        # The trained duration predictor must keep durations at 1, as the app's e2e test expects.
        ref = em.reference_wav(self.fs2, self.vocoder, self.units)
        self.assertEqual(len(ref), self.units.shape[1] * em.HOP + 8)


def has_run(folder):
    return os.path.exists(os.path.join(folder, "decoder_best.pt"))


@unittest.skipUnless(WITH_BIGVGAN, "set WESPER_BIGVGAN=1 to run BigVGAN (and have its checkpoint cached)")
@unittest.skipUnless(all(os.path.exists(p) for p in CACHED), "WESPER's checkpoints aren't cached yet")
@unittest.skipUnless(has_run(SV_BIGVGAN_DECODER) or has_run(SV_DECODER), "no narrator decoder run in decoder/runs")
class BigVGANDecoder(unittest.TestCase):
    """A decoder trained for BigVGAN (or the stand-in): units moved to BigVGAN's frames, FastSpeech2
    and BigVGAN in one graph, against units2wav()."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        run = SV_BIGVGAN_DECODER if has_run(SV_BIGVGAN_DECODER) else stand_in_bigvgan_run(os.path.join(cls.tmp.name, "run"))
        cls.configs = em.load_run_configs(run, em.load_configs()[1])
        cls.voc = em.run_vocoder(cls.configs[0])
        cls.fs2 = em.wn.load_fastspeech2(cls.configs, checkpoint_path=os.path.join(run, "decoder_best.pt"), device="cpu")
        cls.vocoder = em.vocoders.load(cls.voc)
        torch.manual_seed(0)
        cls.mel = torch.randn(1, cls.voc.n_mels, 30) - 5
        cls.before = cls.vocoder(cls.mel)
        cls.filters = em.fixed_bigvgan_filters(cls.vocoder, cls.voc.n_mels)
        wav, _ = sf.read(os.path.join(REPO, "sample_whisper.wav"), dtype="float32")
        cls.units = em.reference_units(em.wn.load_hubert(CACHED[0], device="cpu"), wav)
        cls.ref = em.reference_wav(cls.fs2, cls.vocoder, cls.units, cls.voc)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fixed_filters_leave_the_output_unchanged(self):
        self.assertGreater(self.filters, 100)
        torch.testing.assert_close(self.vocoder(self.mel), self.before, rtol=0, atol=1e-6)

    def test_reference_wav_is_units2wav(self):
        gui, _ = em.wn.units2wav(self.units, self.fs2, self.vocoder, self.configs[1], self.configs[0], device="cpu",
                                 vocoder_spec=self.voc)
        np.testing.assert_array_equal((np.clip(self.ref.numpy(), -1, 1) * 32767).astype("int16"), gui)

    def test_one_frame_per_bigvgan_frame(self):
        # The duration predictor keeps durations at 1, as the app's e2e test expects.
        self.assertEqual(len(self.ref), em.vocoders.n_frames(self.units.shape[1], self.voc) * self.voc.hop)

    def test_decoder_wrapper_matches_reference(self):
        wav = em.DecoderExport(self.fs2, self.vocoder, self.voc).eval()(self.units)[0]
        self.assertEqual(len(wav), len(self.ref))
        self.assertGreater(em.snr_db(self.ref, wav), em.MIN_SNR_DB)

    def test_onnx_export_matches_reference(self):
        # export_decoder checks the file against reference_wav on each input, and raises if they disagree
        units = [self.units.numpy(), self.units[:, :57].numpy()]
        info, checks = em.export_decoder(self.fs2, self.vocoder, "test", self.tmp.name, units, lambda s: None, self.voc)
        self.assertEqual(len(checks), 2)
        self.assertGreater(info["bytes"], 400e6)


@unittest.skipUnless(os.environ.get("WESPER_EXPORT_SMOKE"), "set WESPER_EXPORT_SMOKE=1 to run the full export")
@unittest.skipUnless(all(os.path.exists(p) for p in CACHED), "WESPER's checkpoints aren't cached yet")
class ExportSmoke(unittest.TestCase):
    def test_full_export(self):
        has_sv, has_sv_decoder = os.path.exists(SV_ENCODER), has_run(SV_DECODER)
        with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory() as runs:
            args = [sys.executable, os.path.join(REPO, "web", "export_models.py"), "--out", out, "--timeout", "1500"]
            args += ["--original", CACHED[0], "--fastspeech2", CACHED[1], "--hifigan", CACHED[2]]
            if not has_sv:
                args += ["--sv", ""]
            bigvgan_run = SV_BIGVGAN_DECODER if has_run(SV_BIGVGAN_DECODER) else None
            if not bigvgan_run and WITH_BIGVGAN and has_sv_decoder:
                bigvgan_run = stand_in_bigvgan_run(os.path.join(runs, "run"))
            args += ["--sv-bigvgan-decoder", bigvgan_run or ""]
            r = subprocess.run(args, capture_output=True, text=True, timeout=1600, cwd=tempfile.gettempdir())
            self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])

            manifest = json.load(open(os.path.join(out, "models.json")))
            self.assertEqual([e["id"] for e in manifest["encoders"]], ["sv", "original"] if has_sv else ["original"])
            self.assertEqual((manifest["version"], manifest["sampleRate"], manifest["hop"]), (2, 16000, 320))
            original = manifest["encoders"][-1]
            self.assertEqual((original["targetDbfs"], original["maxGainDb"]), (None, None))
            if has_sv:
                self.assertEqual((manifest["encoders"][0]["targetDbfs"], manifest["encoders"][0]["maxGainDb"]), (-20.0, 40.0))
            expected = ["sv-narrator"] * has_sv_decoder + ["sv-narrator-bigvgan"] * bool(bigvgan_run) + ["googletts"]
            self.assertEqual([d["id"] for d in manifest["decoders"]], expected)
            for d in manifest["decoders"]:
                self.assertEqual((d["vocoder"], d["sampleRate"], d["hop"]),
                                 ("bigvgan22k", 22050, 256) if d["id"] == "sv-narrator-bigvgan" else ("hifigan16k", 16000, 320))
            self.assertEqual(sorted(f for f in os.listdir(out) if f.endswith(".onnx")),
                             sorted([f"encoder-{e['id']}.onnx" for e in manifest["encoders"]] +
                                    [f"decoder-{d['id']}.onnx" for d in manifest["decoders"]]))
            for entry in manifest["encoders"] + manifest["decoders"]:
                info = entry["file"]
                path = os.path.join(out, info["path"])
                self.assertEqual(os.path.getsize(path), info["bytes"])
                self.assertEqual(hashlib.sha256(open(path, "rb").read()).hexdigest(), info["sha256"])
                self.assertEqual(len(entry["checks"]), 2)  # both test clips


if __name__ == "__main__":
    unittest.main()
