"""Tests for colab/prepare_data.py.

Run from the repo root: .venv/bin/python -m unittest discover -s colab/tests -v
The end-to-end tests need ffmpeg on PATH.
"""
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest

import numpy as np
import soundfile as sf

COLAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COLAB_DIR)

import prepare_data as pd  # noqa: E402

SR = pd.SR
rng = np.random.default_rng(0)


def noise(seconds, dbfs):
    """Stationary white noise whose RMS is dbfs, a stand-in for speech in level tests."""
    return (rng.standard_normal(int(seconds * SR)) * 10 ** (dbfs / 20)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


def thud(seconds=0.03, dbfs=-30):
    """A short burst that decays exponentially, like a touchpad click."""
    x = noise(seconds, dbfs)
    return (x * np.exp(-np.linspace(0, 6, len(x)))).astype(np.float32)


def clip_with_thud(level=-20):
    """MP3-like leading zeros, a thud, quiet room noise, then speech. Returns audio and speech start."""
    parts = [silence(0.04), thud(dbfs=level - 10), noise(0.3, level - 65)]
    speech_start = sum(len(p) for p in parts)
    return np.concatenate(parts + [noise(1.0, level)]), speech_start


class SpeechLevel(unittest.TestCase):
    def test_matches_rms_of_steady_signal(self):
        self.assertAlmostEqual(pd.speech_dbfs(noise(1, -23)), -23, delta=0.5)

    def test_follows_gain_exactly(self):
        x = noise(1, -30)
        self.assertAlmostEqual(pd.speech_dbfs(x * 10 ** (12 / 20)) - pd.speech_dbfs(x), 12, places=3)

    def test_ignores_leading_and_trailing_silence(self):
        x = noise(1, -20)
        padded = np.concatenate([silence(2), x, silence(1)])
        self.assertAlmostEqual(pd.speech_dbfs(padded), pd.speech_dbfs(x), delta=0.1)

    def test_ignores_a_click(self):
        x = noise(2, -30)
        clicked = x.copy()
        clicked[SR // 2:SR // 2 + 40] = 0.95
        self.assertAlmostEqual(pd.speech_dbfs(clicked), pd.speech_dbfs(x), delta=1.0)

    def test_ignores_quiet_background_noise(self):
        x = np.concatenate([silence(0.5), noise(1, -20), silence(0.5)])
        noisy = x + noise(len(x) / SR, -65)
        self.assertAlmostEqual(pd.speech_dbfs(noisy), pd.speech_dbfs(x), delta=0.5)

    def test_silence_and_empty_input(self):
        self.assertEqual(pd.speech_dbfs(silence(1)), -120.0)
        self.assertEqual(pd.speech_dbfs(np.zeros(10, dtype=np.float32)), -120.0)


class ThudEnd(unittest.TestCase):
    def test_finds_thud_before_quiet_gap(self):
        x, speech_start = clip_with_thud()
        cut = pd.thud_end(x, pd.speech_dbfs(x))
        thud_end_sample = int((0.04 + 0.03) * SR)
        self.assertGreaterEqual(cut, thud_end_sample - SR // 100)  # within 10 ms of the burst's end
        self.assertLess(cut, thud_end_sample + int(0.05 * SR))
        self.assertLess(cut, speech_start)

    def test_speech_right_at_start_is_kept(self):
        x = np.concatenate([silence(0.04), noise(1, -20)])
        self.assertEqual(pd.thud_end(x, pd.speech_dbfs(x)), 0)

    def test_quiet_start_has_nothing_to_mute(self):
        x = np.concatenate([silence(0.04), noise(0.3, -85), noise(1, -20)])
        self.assertEqual(pd.thud_end(x, pd.speech_dbfs(x)), 0)

    def test_sound_longer_than_thud_max_is_kept(self):
        long_sound = noise((pd.THUD_MAX_MS + 100) / 1000, -30)
        x = np.concatenate([silence(0.04), long_sound, noise(0.3, -85), noise(1, -20)])
        self.assertEqual(pd.thud_end(x, pd.speech_dbfs(x)), 0)

    def test_thud_without_quiet_gap_is_kept(self):
        x = np.concatenate([silence(0.04), thud(dbfs=-30), noise(1, -20)])
        self.assertEqual(pd.thud_end(x, pd.speech_dbfs(x)), 0)

    def test_all_zeros(self):
        self.assertEqual(pd.thud_end(silence(1), -120.0), 0)


class Process(unittest.TestCase):
    STORED = pd.TARGET_DBFS - pd.HEADROOM_DB

    def test_both_versions_reach_stored_level(self):
        for level in (-40, -20, -8):
            normal, whisper, _ = pd.process(noise(1, level), noise(1, level - 17))
            self.assertAlmostEqual(pd.speech_dbfs(normal), self.STORED, delta=0.3, msg=f"normal from {level}")
            self.assertAlmostEqual(pd.speech_dbfs(whisper), self.STORED, delta=0.3, msg=f"whisper from {level}")

    def test_trims_to_common_length(self):
        normal, whisper, _ = pd.process(noise(1, -20), noise(1.005, -35))
        self.assertEqual(len(normal), SR)
        self.assertEqual(len(whisper), SR)

    def test_mutes_thud_in_both_versions_and_keeps_alignment(self):
        x, speech_start = clip_with_thud()
        whisper = noise(len(x) / SR, -40)
        marker = speech_start + SR // 2
        whisper[marker] = 0.9  # a spike that must stay at the same sample
        normal_out, whisper_out, info = pd.process(x, whisper)
        cut = int(info["thud_ms"] * SR / 1000)
        self.assertGreater(cut, 0)
        self.assertTrue(np.all(normal_out[:cut] == 0))
        self.assertTrue(np.all(whisper_out[:cut] == 0))
        self.assertTrue(np.any(normal_out[speech_start:] != 0))
        self.assertEqual(int(np.argmax(np.abs(whisper_out))), marker)

    def test_gain_is_capped(self):
        normal, _, _ = pd.process(noise(1, -75), noise(1, -20))
        self.assertAlmostEqual(pd.speech_dbfs(normal), -75 + pd.MAX_GAIN_DB - pd.HEADROOM_DB, delta=0.5)

    def test_reports_levels_before_processing(self):
        _, _, info = pd.process(noise(1, -31), noise(1, -48))
        self.assertAlmostEqual(info["normal_dbfs"], -31, delta=0.5)
        self.assertAlmostEqual(info["whisper_dbfs"], -48, delta=0.5)

    def test_clipping_is_counted_and_bounded(self):
        spikes = noise(1, -50)
        spikes[::8000] = 1.0  # rare full-scale spikes on quiet audio: the level gain pushes them past 1.0
        normal, _, info = pd.process(spikes, noise(1, -30))
        self.assertGreater(info["clipped"], 0)
        self.assertLessEqual(np.abs(normal).max(), 1.0)

    def test_does_not_modify_inputs(self):
        x, _ = clip_with_thud()
        w = noise(len(x) / SR, -40)
        x0, w0 = x.copy(), w.copy()
        pd.process(x, w)
        np.testing.assert_array_equal(x, x0)
        np.testing.assert_array_equal(w, w0)


class Helpers(unittest.TestCase):
    def test_flac_round_trip_only_quantizes_to_16_bit(self):
        x = np.clip(noise(0.5, -20), -1, 1)
        y, sr = sf.read(io.BytesIO(pd.flac_bytes(x)), dtype="float32")
        self.assertEqual(sr, SR)
        self.assertLessEqual(np.abs(y - x).max(), 1.01 / 32768)  # within one 16-bit step

    def test_flac_is_lossless_for_16_bit_audio(self):
        x = np.round(noise(0.5, -20) * 32768).clip(-32768, 32767) / 32768
        y, _ = sf.read(io.BytesIO(pd.flac_bytes(x.astype(np.float32))), dtype="float32")
        np.testing.assert_array_equal(y, x.astype(np.float32))

    def test_pick_val_cycles_through_speakers(self):
        rows = [{"client_id": s, "path": f"{s}{i}.mp3"} for s, n in (("a", 10), ("b", 3), ("c", 1), ("d", 1)) for i in range(n)]
        picked = pd.pick_val(rows, 7, __import__("random").Random(0))
        self.assertEqual(len({r["path"] for r in picked}), 7)
        counts = sorted((sum(r["client_id"] == s for r in picked) for s in "abcd"), reverse=True)
        self.assertEqual(counts, [3, 2, 1, 1])

    def test_pick_val_is_deterministic(self):
        rows = [{"client_id": f"s{i % 5}", "path": f"{i}.mp3"} for i in range(40)]
        random = __import__("random")
        self.assertEqual(pd.pick_val(rows, 10, random.Random(3)), pd.pick_val(rows, 10, random.Random(3)))


def _tree_digest(root):
    """Hash of every file's path and contents under root."""
    h = hashlib.sha256()
    for dirpath, _, files in sorted(os.walk(root)):
        for name in sorted(files):
            path = os.path.join(dirpath, name)
            h.update(os.path.relpath(path, root).encode())
            with open(path, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


# (clip, speaker, kind) for the fake corpus. Speakers t1/t2 are in test.tsv, s1/s2 are not.
FAKE_CLIPS = [
    ("c_s1_a", "s1", "thud"), ("c_s1_b", "s1", "speech"), ("c_s1_quiet", "s1", "quiet"),
    ("c_s2_a", "s2", "speech"), ("c_s2_b", "s2", "thud"),
    ("c_t1_a", "t1", "speech"), ("c_t1_b", "t1", "speech"), ("c_t2_a", "t2", "thud"),
]


def make_fake_corpus(root):
    """Write a tiny Common Voice-style corpus and matching pseudo-whispers under root.

    Returns (corpus_dir, n2w_dir). Needs ffmpeg, to write the clips as mp3 like Common Voice.
    """
    corpus, n2w = os.path.join(root, "corpus"), os.path.join(root, "n2w")
    os.makedirs(os.path.join(corpus, "clips"))
    os.makedirs(n2w)
    header = "client_id\tpath\tsentence\n"
    validated, test, durations = [header], [header], ["clip\tduration[ms]\n"]
    for name, speaker, kind in FAKE_CLIPS:
        if kind == "thud":
            x, _ = clip_with_thud(-18)
        elif kind == "quiet":
            x = np.concatenate([silence(0.2), noise(1, -60)])
        else:
            x = np.concatenate([silence(0.04), noise(1, -25)])
        mp3 = os.path.join(corpus, "clips", name + ".mp3")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1",
                        "-i", "pipe:0", "-ar", "48000", mp3], input=x.astype(np.float32).tobytes(), check=True,
                       timeout=60)
        # The pseudo-whisper is a little longer, like Normal2Whisper's output.
        sf.write(os.path.join(n2w, name + ".wav"), noise(len(x) / SR + 0.005, -40), SR, subtype="PCM_16")
        line = f"{speaker}\t{name}.mp3\tSentence for {name}.\n"
        validated.append(line)
        if speaker.startswith("t"):
            test.append(line)
        durations.append(f"{name}.mp3\t{len(x) * 1000 // SR}\n")
    for fname, lines in (("validated.tsv", validated), ("test.tsv", test), ("clip_durations.tsv", durations)):
        with open(os.path.join(corpus, fname), "w") as f:
            f.writelines(lines)
    return corpus, n2w


class EndToEnd(unittest.TestCase):
    """Runs the script on a tiny synthetic corpus laid out like Common Voice."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = cls.tmp.name
        cls.clips = FAKE_CLIPS
        cls.corpus, cls.n2w = make_fake_corpus(root)

        cls.digest_before = (_tree_digest(cls.corpus), _tree_digest(cls.n2w))
        cls.out = os.path.join(root, "out", "data.tar")
        cls.export = os.path.join(root, "out", "export")
        cls.result = cls.run_script(cls.out, "--export-dir", cls.export, "--val-clips", "2")
        cls.tar = tarfile.open(cls.out)
        cls.manifest = list(csv.DictReader(io.StringIO(cls.tar.extractfile("manifest.tsv").read().decode()),
                                           delimiter="\t", quoting=csv.QUOTE_NONE))

    @classmethod
    def tearDownClass(cls):
        cls.tar.close()
        cls.tmp.cleanup()

    @classmethod
    def run_script(cls, out, *extra, n2w=None):
        return subprocess.run([sys.executable, os.path.join(COLAB_DIR, "prepare_data.py"), cls.corpus,
                               n2w or cls.n2w, out, "--jobs", "2", *extra], capture_output=True, text=True,
                              timeout=300)

    def test_succeeds(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr)

    def test_near_silent_clip_is_skipped(self):
        self.assertNotIn("c_s1_quiet", {r["clip"] for r in self.manifest})
        self.assertIn("skipped 1 near-silent", self.result.stdout)

    def test_splits_are_speaker_disjoint(self):
        train = {r["speaker"] for r in self.manifest if r["split"] == "train"}
        val = {r["speaker"] for r in self.manifest if r["split"] == "val"}
        self.assertEqual(train, {"s1", "s2"})
        self.assertEqual(val, {"t1", "t2"})  # --val-clips 2 takes one clip per test speaker
        self.assertEqual(len([r for r in self.manifest if r["split"] == "val"]), 2)

    def test_tar_has_a_pair_per_clip_plus_metadata(self):
        names = set(self.tar.getnames())
        self.assertIn("manifest.tsv", names)
        self.assertIn("prep.json", names)
        for r in self.manifest:
            self.assertIn(f"normal/{r['clip']}.flac", names)
            self.assertIn(f"whisper/{r['clip']}.flac", names)
        self.assertEqual(len(names), 2 * len(self.manifest) + 2)

    def test_pairs_are_aligned_and_at_stored_level(self):
        for r in self.manifest:
            normal, sr = sf.read(io.BytesIO(self.tar.extractfile(f"normal/{r['clip']}.flac").read()))
            whisper, _ = sf.read(io.BytesIO(self.tar.extractfile(f"whisper/{r['clip']}.flac").read()))
            self.assertEqual(sr, SR)
            self.assertEqual(len(normal), len(whisper), r["clip"])
            self.assertAlmostEqual(pd.speech_dbfs(normal), pd.TARGET_DBFS - pd.HEADROOM_DB, delta=0.5, msg=r["clip"])
            self.assertAlmostEqual(pd.speech_dbfs(whisper), pd.TARGET_DBFS - pd.HEADROOM_DB, delta=0.5, msg=r["clip"])

    def test_thuds_are_muted_and_speech_starts_are_not(self):
        kinds = {name: kind for name, _, kind in self.clips}
        for r in self.manifest:
            if kinds[r["clip"]] == "thud":
                self.assertGreater(float(r["thud_ms"]), 0, r["clip"])
            else:
                self.assertEqual(float(r["thud_ms"]), 0, r["clip"])

    def test_prep_json_records_settings(self):
        prep = json.loads(self.tar.extractfile("prep.json").read())
        self.assertEqual(prep["target_dbfs"], pd.TARGET_DBFS)
        self.assertEqual(prep["headroom_db"], pd.HEADROOM_DB)
        self.assertEqual(prep["sample_rate"], SR)

    def test_export_matches_tar_byte_for_byte(self):
        for member in self.tar.getmembers():
            with open(os.path.join(self.export, member.name), "rb") as f:
                self.assertEqual(f.read(), self.tar.extractfile(member).read(), member.name)

    def test_sources_are_untouched(self):
        self.assertEqual((_tree_digest(self.corpus), _tree_digest(self.n2w)), self.digest_before)

    def test_refuses_output_inside_sources(self):
        for inside in (os.path.join(self.corpus, "x.tar"), os.path.join(self.n2w, "x.tar")):
            result = self.run_script(inside)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(os.path.exists(inside) or os.path.exists(inside + ".partial"))
        self.assertEqual((_tree_digest(self.corpus), _tree_digest(self.n2w)), self.digest_before)

    def test_missing_pseudo_whisper_fails_without_writing(self):
        partial_n2w = os.path.join(self.tmp.name, "n2w_partial")
        os.makedirs(partial_n2w, exist_ok=True)
        out = os.path.join(self.tmp.name, "missing.tar")
        result = self.run_script(out, n2w=partial_n2w)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no Normal2Whisper file", result.stderr)
        self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
