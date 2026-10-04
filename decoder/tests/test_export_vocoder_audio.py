"""Tests for decoder/export_vocoder_audio.py.

Fake recordings are prepared for BigVGAN's settings (with a stand-in for the encoder, so nothing
is downloaded), exported at 22.05 kHz, and the exported audio's mel is checked against the
targets prepare_data.py stored.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(TESTS))
sys.path.insert(0, REPO)
sys.path.insert(0, TESTS)

import vocoders  # noqa: E402
from decoder import export_vocoder_audio as eva, prepare_data as pdd  # noqa: E402
from test_prepare_data import make_recordings, noise  # noqa: E402

# bigvgan22k's settings (vocoders.spec would download its config.json)
BIG = vocoders.Spec("bigvgan22k", 22050, 1024, 256, 1024, 80, 0, 8000, "")
SCRIPT = os.path.join(REPO, "decoder", "export_vocoder_audio.py")


class FakeEncoder:
    def units(self, x):
        return torch.zeros(1, x.shape[-1] // pdd.HOP, 256)


def prepare(src, out, voc, val=()):
    """prepare_data.py's output for the recordings in src, as its main() writes it; `val` names the validation recordings."""
    for sub in ("segments", "audio"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    rows = []
    for name in sorted(os.listdir(src)):
        rows += pdd.prepare_recording(os.path.join(src, name), out, FakeEncoder(), "cpu", voc)
    rows = [{**r, "split": "val" if r["recording"] in val else "train"} for r in rows]
    with open(os.path.join(out, "segments.tsv"), "w", newline="") as f:
        writer = csv.DictWriter(f, pdd.FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(out, "prep.json"), "w") as f:
        json.dump({"vocoder": voc.name, "sample_rate": voc.sample_rate, "hop": voc.hop}, f)
    segments = [np.load(os.path.join(out, "segments", r["id"] + ".npz")) for r in rows]
    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(pdd.compute_stats([d["pitch"] for d in segments], [d["energy"] for d in segments]), f)
    return rows


def export(*args):
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True, timeout=300)


class Cut(unittest.TestCase):
    def test_at_16_khz_it_is_the_normalized_utterance(self):
        x = noise(3, -30)
        row = {"id": "a", "start": "0.40", "end": "2.20", "frames": "85"}
        expected, _ = pdd.normalize(x[6400:35200])
        np.testing.assert_array_equal(eva.cut(x, x, row, 16000, 320), expected[:85 * 320])

    def test_too_few_samples_for_the_frames_is_an_error(self):
        x = noise(1, -30)
        with self.assertRaises(ValueError):
            eva.cut(x, x, {"id": "a", "start": "0.00", "end": "1.00", "frames": "51"}, 16000, 320)


class Export(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.src = os.path.join(cls.tmp.name, "book")
        make_recordings(cls.src, n=2)
        cls.data = os.path.join(cls.tmp.name, "data")
        cls.rows = prepare(cls.src, cls.data, BIG)
        cls.out = os.path.join(cls.tmp.name, "audio22k")
        cls.result = export(cls.src, cls.data, cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_succeeds(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr[-2000:])
        self.assertGreaterEqual(len(self.rows), 4)
        self.assertIn(f"exported {len(self.rows)} utterances", self.result.stdout)

    def test_one_file_per_utterance_at_the_vocoders_rate_and_frames(self):
        for r in self.rows:
            info = sf.info(os.path.join(self.out, r["id"] + ".flac"))
            self.assertEqual(info.samplerate, 22050)
            self.assertEqual(info.frames, int(r["frames"]) * 256)

    def test_its_mel_is_the_stored_target(self):
        for r in self.rows:
            wav, _ = eva.load(os.path.join(self.out, r["id"] + ".flac"))
            mel, _ = vocoders.mel_energy(wav, BIG)
            with np.load(os.path.join(self.data, "segments", r["id"] + ".npz")) as d:
                target = d["mel"].astype(np.float32)
            self.assertEqual(mel.shape, target.shape)
            # All but the last 2 frames, whose analysis windows reach past the cut.
            np.testing.assert_allclose(mel[:, :-2], target[:, :-2], atol=0.02)

    def test_records_its_settings(self):
        with open(os.path.join(self.out, "export.json")) as f:
            self.assertEqual(json.load(f), {"vocoder": "bigvgan22k", "sample_rate": 22050, "hop": 256,
                                             "headroom_db": eva.HEADROOM_DB, "utterances": len(self.rows)})

    def test_rerun_skips_exported_utterances(self):
        first = os.path.join(self.out, self.rows[0]["id"] + ".flac")
        before = os.stat(first).st_mtime_ns
        result = export(self.src, self.data, self.out)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn(f"exported 0 utterances ({len(self.rows)} already there)", result.stdout)
        self.assertEqual(os.stat(first).st_mtime_ns, before)

    def test_missing_recording_is_an_error(self):
        empty = os.path.join(self.tmp.name, "empty")
        os.makedirs(empty, exist_ok=True)
        result = export(empty, self.data, os.path.join(self.tmp.name, "other"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no recording", result.stderr)


if __name__ == "__main__":
    unittest.main()
