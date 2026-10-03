"""Tests for decoder/prepare_data.py.

The end-to-end test runs WESPER's encoder and is opt-in, because it needs the encoder checkpoint
(cached in torch's hub directory after any WESPER run):

    WESPER_DECODER_SMOKE=1 .venv/bin/python -m unittest discover -s decoder/tests -v
"""
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(TESTS))
sys.path.insert(0, REPO)

from decoder import prepare_data as pdd  # noqa: E402
from colab.prepare_data import speech_dbfs  # noqa: E402

SR, HOP = pdd.SR, pdd.HOP
ENCODER_CACHE = os.path.expanduser("~/.cache/torch/hub/checkpoints/model-layer12-450000.pt")


def noise(seconds, dbfs, seed=0):
    return (np.random.default_rng(seed).standard_normal(int(seconds * SR)) * 10 ** (dbfs / 20)).astype(np.float32)


def tone(seconds, hz, dbfs=-20):
    t = np.arange(int(seconds * SR)) / SR
    x = sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, 15))
    return (x / np.sqrt(np.mean(x ** 2)) * 10 ** (dbfs / 20)).astype(np.float32)


def fake_chapter(bursts, pause=0.5, seed=0):
    """Speech-like bursts (seconds each) separated by quiet pauses. Returns audio and burst spans."""
    parts, spans, pos = [noise(0.3, -75, seed)], [], int(0.3 * SR)
    for i, b in enumerate(bursts):
        x = noise(b, -20, seed + i + 1)
        spans.append((pos, pos + len(x)))
        parts += [x, noise(pause, -75, seed + 100 + i)]
        pos += len(x) + int(pause * SR)
    return np.concatenate(parts), spans


class MelEnergy(unittest.TestCase):
    def test_one_frame_per_unit(self):
        for n in (16000, 16320, 16639, 50000):
            mel, energy = pdd.mel_energy(noise(n / SR, -20))
            self.assertEqual(mel.shape, (80, n // HOP))
            self.assertEqual(energy.shape, (n // HOP,))

    def test_matches_hifigan_meldataset(self):
        # HiFi-GAN's meldataset.mel_spectrogram, reimplemented with librosa (not torch.stft).
        import librosa
        x = noise(1.5, -20)
        padded = np.pad(x, ((1024 - HOP) // 2, (1024 - HOP) // 2), mode="reflect")
        spec = librosa.stft(padded, n_fft=1024, hop_length=HOP, win_length=1024, window="hann", center=False)
        mag = np.sqrt(np.abs(spec) ** 2 + 1e-9)
        basis = librosa.filters.mel(sr=SR, n_fft=1024, n_mels=80, fmin=0, fmax=8000)
        expected = np.log(np.clip(basis @ mag, 1e-5, None))
        mel, _ = pdd.mel_energy(x)
        np.testing.assert_allclose(mel, expected, atol=1e-4)

    def test_frames_are_centered_like_units(self):
        # Frame t covers samples centered on t * HOP + HOP / 2, so a click there peaks in frame t.
        for t in (5, 20, 33):
            x = np.zeros(SR, np.float32)
            x[t * HOP + HOP // 2] = 1.0
            _, energy = pdd.mel_energy(x)
            self.assertEqual(int(np.argmax(energy)), t)


class Pitch(unittest.TestCase):
    def test_tracks_a_voiced_tone(self):
        f0 = pdd.pitch(tone(1.0, 180), 50)
        self.assertEqual(len(f0), 50)
        self.assertAlmostEqual(float(np.median(f0)), 180, delta=4)

    def test_fills_unvoiced_gaps(self):
        x = np.concatenate([tone(0.5, 150), np.zeros(SR // 2, np.float32), tone(0.5, 150)])
        f0 = pdd.pitch(x, len(x) // HOP)
        self.assertTrue((f0 > 0).all())

    def test_silence_gives_zeros(self):
        self.assertTrue((pdd.pitch(np.zeros(SR, np.float32), 50) == 0).all())


class Segment(unittest.TestCase):
    def assert_valid(self, segments, n):
        for (s, e), nxt in zip(segments, segments[1:] + [(n, n)]):
            self.assertGreaterEqual((e - s) / SR, pdd.MIN_SECONDS)
            self.assertLessEqual((e - s) / SR, pdd.MAX_SECONDS + 2 * pdd.EDGE_SECONDS)
            self.assertLessEqual(e, nxt[0])

    def test_cuts_only_in_pauses(self):
        x, spans = fake_chapter([3, 2, 4, 6, 5, 2.5])
        segments = pdd.segment(x)
        self.assert_valid(segments, len(x))
        for bs, be in spans:  # every burst lies wholly inside one utterance
            self.assertEqual(sum(s <= bs and be <= e for s, e in segments), 1, (bs, be, segments))

    def test_keeps_a_little_silence_at_the_edges(self):
        x, spans = fake_chapter([3])
        (s, e), = pdd.segment(x)
        edge = pdd.EDGE_SECONDS * SR
        self.assertLessEqual(abs((spans[0][0] - s) - edge), HOP)
        self.assertLessEqual(abs((e - spans[0][1]) - edge), HOP)

    def test_splits_long_speech_without_pauses(self):
        x, _ = fake_chapter([30])
        segments = pdd.segment(x)
        self.assertGreaterEqual(len(segments), 3)
        self.assert_valid(segments, len(x))

    def test_silence_gives_nothing(self):
        self.assertEqual(pdd.segment(noise(5, -80)), [])


class Helpers(unittest.TestCase):
    def test_normalize_reaches_target(self):
        y, gain = pdd.normalize(noise(2, -35))
        self.assertAlmostEqual(speech_dbfs(y), pdd.TARGET_DBFS, delta=0.1)
        self.assertAlmostEqual(gain, 15, delta=0.5)

    def test_stats_in_fastspeech2_format(self):
        p = [np.array([100.0, 150.0, 200.0]), np.array([120.0, 180.0, 150.0])]
        stats = pdd.compute_stats(p, [np.array([1.0, 3.0]), np.array([2.0, 2.0])])
        values = np.concatenate(p)
        mean, std = values.mean(), values.std()
        np.testing.assert_allclose(stats["pitch"], [(100 - mean) / std, (200 - mean) / std, mean, std])

    def test_stats_fit_ignores_outliers_but_range_includes_them(self):
        # One octave error at 520 Hz in an otherwise steady utterance, as pitch trackers make.
        utterance = np.array([130.0, 132, 128, 131, 129, 130, 520])
        stats = pdd.compute_stats([utterance], [np.ones(3)])
        mean, std = utterance[:-1].mean(), utterance[:-1].std()
        np.testing.assert_allclose(stats["pitch"][2:], [mean, std])
        np.testing.assert_allclose(stats["pitch"][1], (520 - mean) / std)

    def test_remove_outliers_matches_fastspeech2(self):
        np.testing.assert_array_equal(pdd.remove_outliers(np.array([1.0, 2, 3, 4, 100])), [1, 2, 3, 4])


def tree_digest(root):
    h = hashlib.sha256()
    for dirpath, _, files in sorted(os.walk(root)):
        for name in sorted(files):
            with open(os.path.join(dirpath, name), "rb") as f:
                h.update(name.encode() + f.read())
    return h.hexdigest()


def make_recordings(folder, n=3):
    """n short fake 'chapters' as mp3, like an audiobook folder."""
    os.makedirs(folder)
    for i in range(n):
        x, _ = fake_chapter([3, 4, 2.5], seed=10 * i)
        x = np.concatenate([x, tone(3, 140 + 20 * i), noise(0.5, -75)])
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
                        "-ar", "44100", os.path.join(folder, f"chapter_{i:03d}.mp3")],
                       input=x.tobytes(), check=True, timeout=60)


@unittest.skipUnless(os.environ.get("WESPER_DECODER_SMOKE"), "set WESPER_DECODER_SMOKE=1 to run WESPER's encoder")
class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(ENCODER_CACHE):
            raise unittest.SkipTest("WESPER's encoder isn't cached yet")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.src = os.path.join(cls.tmp.name, "book")
        make_recordings(cls.src)
        cls.digest = tree_digest(cls.src)
        cls.out = os.path.join(cls.tmp.name, "data")
        cls.result = cls.run_script(cls.src, cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def run_script(src, out, *extra):
        return subprocess.run([sys.executable, os.path.join(REPO, "decoder", "prepare_data.py"), src, out,
                               "--val-recordings", "1", *extra], capture_output=True, text=True, timeout=600)

    def rows(self):
        with open(os.path.join(self.out, "segments.tsv"), newline="") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    def test_succeeds_and_splits_by_recording(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr[-2000:])
        rows = self.rows()
        self.assertEqual({r["recording"] for r in rows if r["split"] == "val"}, {"chapter_002"})
        self.assertEqual({r["recording"] for r in rows if r["split"] == "train"}, {"chapter_000", "chapter_001"})

    def test_features_are_aligned(self):
        for r in self.rows():
            with np.load(os.path.join(self.out, "segments", r["id"] + ".npz")) as d:
                n = int(r["frames"])
                self.assertEqual(d["units"].shape, (n, 256))
                self.assertEqual(d["mel"].shape, (80, n))
                self.assertEqual(d["pitch"].shape, (n,))
                self.assertEqual(d["energy"].shape, (n,))
            self.assertTrue(os.path.exists(os.path.join(self.out, "audio", r["id"] + ".flac")))

    def test_writes_stats_and_settings(self):
        with open(os.path.join(self.out, "stats.json")) as f:
            stats = json.load(f)
        self.assertEqual(set(stats), {"pitch", "energy"})
        self.assertEqual(len(stats["pitch"]), 4)
        with open(os.path.join(self.out, "prep.json")) as f:
            self.assertEqual(json.load(f)["mel"], "hifigan")

    def test_rerun_skips_finished_recordings(self):
        npz = os.path.join(self.out, "segments", self.rows()[0]["id"] + ".npz")
        before = os.stat(npz).st_mtime_ns
        self.assertEqual(self.run_script(self.src, self.out).returncode, 0)
        self.assertEqual(os.stat(npz).st_mtime_ns, before)

    def test_source_untouched_and_guarded(self):
        result = self.run_script(self.src, os.path.join(self.src, "inner"))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(tree_digest(self.src), self.digest)


if __name__ == "__main__":
    unittest.main()
