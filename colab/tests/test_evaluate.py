"""Tests for colab/evaluate.py.

The scoring tests use a stand-in recognizer, so they need neither transformers nor a model.
The convert test runs WESPER on a tiny fake dataset and is opt-in, because it needs WESPER's
checkpoints (cached by the notebook smoke test or a first run):

    WESPER_EVAL_SMOKE=1 .venv/bin/python -m unittest discover -s colab/tests -p test_evaluate.py -v
"""
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
COLAB_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, COLAB_DIR)
sys.path.insert(0, TESTS_DIR)

import evaluate as ev  # noqa: E402


class Text(unittest.TestCase):
    def test_normalize_ignores_case_punctuation_and_spacing(self):
        self.assertEqual(ev.normalize_text("Nej, tala  om det för MIG."), "nej tala om det för mig")
        self.assertEqual(ev.normalize_text("Åsa - ja! \"Öl\"?"), "åsa ja öl")
        self.assertEqual(ev.normalize_text("snake_case 42"), "snake case 42")
        self.assertEqual(ev.normalize_text("  ...  "), "")

    def test_edit_distance(self):
        self.assertEqual(ev.edit_distance("kitten", "sitting"), 3)
        self.assertEqual(ev.edit_distance("", "abc"), 3)
        self.assertEqual(ev.edit_distance("abc", ""), 3)
        self.assertEqual(ev.edit_distance("same", "same"), 0)
        self.assertEqual(ev.edit_distance("en två tre".split(), "en tre".split()), 1)

    def test_score_counts_after_normalizing(self):
        s = ev.score("Nej, tala om det.", "nej tala om det")
        self.assertEqual((s["char_edits"], s["word_edits"]), (0, 0))
        s = ev.score("en två tre", "en tva tre")
        self.assertEqual((s["char_edits"], s["chars"], s["word_edits"], s["words"]), (1, 10, 1, 3))

    def test_error_rate_is_micro_averaged(self):
        scores = [{"char_edits": 1, "chars": 10, "word_edits": 1, "words": 2},
                  {"char_edits": 0, "chars": 30, "word_edits": 0, "words": 6}]
        self.assertAlmostEqual(ev.error_rate(scores), 1 / 40)
        self.assertAlmostEqual(ev.error_rate(scores, "word"), 1 / 8)
        self.assertEqual(ev.error_rate([]), 0.0)


class Bootstrap(unittest.TestCase):
    def scores(self, edits):
        return [{"char_edits": e, "chars": 10, "word_edits": 0, "words": 1} for e in edits]

    def test_identical_systems_give_zero_interval(self):
        a = self.scores([1, 2, 3, 0, 5])
        self.assertEqual(ev.paired_bootstrap(a, a), (0.0, 0.0))

    def test_consistently_better_system_has_interval_below_zero(self):
        a, b = self.scores([5, 6, 4, 5, 7, 6]), self.scores([2, 3, 1, 2, 4, 3])
        lo, hi = ev.paired_bootstrap(a, b)
        self.assertLess(hi, 0)
        self.assertLessEqual(lo, hi)

    def test_is_deterministic(self):
        a, b = self.scores([5, 1, 4, 0, 7]), self.scores([2, 3, 1, 2, 4])
        self.assertEqual(ev.paired_bootstrap(a, b, seed=3), ev.paired_bootstrap(a, b, seed=3))


class HeldOut(unittest.TestCase):
    def test_skips_the_selection_clips_and_keeps_only_val(self):
        manifest = ([{"clip": f"t{i}", "split": "train"} for i in range(3)]
                    + [{"clip": f"v{i}", "split": "val"} for i in range(5)])
        self.assertEqual([r["clip"] for r in ev.heldout_rows(manifest, 2)], ["v2", "v3", "v4"])
        self.assertEqual(ev.heldout_rows(manifest, 5), [])


class RunAsr(unittest.TestCase):
    """The scoring stage end to end, with a stand-in recognizer."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = self.tmp.name
        self.refs = [("c1", "s1", "Nej, tala om det för mig."), ("c2", "s2", "En två tre."), ("c3", "s3", "Hej då.")]
        ev.write_tsv(os.path.join(self.out, "reference.tsv"),
                     [{"clip": c, "speaker": s, "sentence": t} for c, s, t in self.refs], ["clip", "speaker", "sentence"])
        self.conditions = ["normal-rec", "whisper-orig", "whisper-ft"]
        for cond in self.conditions:
            os.makedirs(os.path.join(self.out, cond))
            for c, _, _ in self.refs:
                sf.write(os.path.join(self.out, cond, c + ".wav"), np.zeros(1600, np.float32), ev.SR)
        # Transcripts per condition, in the order run_asr asks for them (CONDITIONS order).
        perfect = [t for _, _, t in self.refs]
        self.transcripts = {"normal-rec": perfect, "whisper-orig": ["nej tala", "en", "hej"], "whisper-ft": perfect}
        self.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def transcribe(self, waves):
        cond = self.conditions[len(self.calls)]
        self.calls.append((cond, len(waves)))
        return self.transcripts[cond]

    def test_scores_each_condition_and_compares_whisper_systems(self):
        summary = ev.run_asr(self.out, self.transcribe)
        self.assertEqual(self.calls, [(c, 3) for c in self.conditions])
        self.assertIn("| normal-rec | 0.0% | 0.0% |", summary)
        self.assertIn("| whisper-ft | 0.0% | 0.0% |", summary)
        self.assertIn("better on 3 clips, worse on 0, same on 0", summary)
        self.assertIn("3 speakers not in training", summary)
        rows = ev.read_tsv(os.path.join(self.out, "asr.tsv"))
        self.assertEqual(len(rows), 9)
        with open(os.path.join(self.out, "summary.md")) as f:
            self.assertEqual(f.read(), summary)

    def test_rerun_reuses_cached_transcripts(self):
        first = ev.run_asr(self.out, self.transcribe)
        self.calls.clear()
        self.assertEqual(ev.run_asr(self.out, lambda waves: self.fail("should not transcribe again")), first)

    def test_transcribes_in_chunks_and_keeps_order(self):
        sizes = []

        def numbered(waves):
            sizes.append(len(waves))
            return [f"text {sum(sizes[:-1]) + i}" for i in range(len(waves))]
        refs = ev.read_tsv(os.path.join(self.out, "reference.tsv"))
        hyps = ev.transcribe_condition(self.out, "whisper-ft", refs, numbered, chunk=2)
        self.assertEqual(sizes, [2, 1])
        self.assertEqual(hyps, ["text 0", "text 1", "text 2"])

    def test_reports_the_baseline_error_rate(self):
        ev.run_asr(self.out, self.transcribe)
        rows = [r for r in ev.read_tsv(os.path.join(self.out, "asr.tsv")) if r["condition"] == "whisper-orig"]
        edits = sum(int(r["char_edits"]) for r in rows)
        chars = sum(int(r["chars"]) for r in rows)
        self.assertGreater(edits, 0)
        self.assertEqual(chars, sum(len(ev.normalize_text(t)) for _, _, t in self.refs))


ORIGINAL = os.path.expanduser("~/.cache/torch/hub/checkpoints/model-layer12-450000.pt")


@unittest.skipUnless(os.environ.get("WESPER_EVAL_SMOKE"), "set WESPER_EVAL_SMOKE=1 to run WESPER")
class ConvertStage(unittest.TestCase):
    """Uses the original encoder as the 'fine-tuned' one, so the expected distances are known."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(ORIGINAL):
            raise unittest.SkipTest("WESPER's encoder isn't cached yet")
        from test_prepare_data import make_fake_corpus
        cls.tmp = tempfile.TemporaryDirectory()
        corpus, n2w = make_fake_corpus(cls.tmp.name)
        export = os.path.join(cls.tmp.name, "export")
        subprocess.run([sys.executable, os.path.join(COLAB_DIR, "prepare_data.py"), corpus, n2w,
                        os.path.join(cls.tmp.name, "data.tar"), "--export-dir", export, "--val-clips", "2"],
                       check=True, capture_output=True, timeout=300)
        cls.out = os.path.join(cls.tmp.name, "eval")
        cls.result = subprocess.run([sys.executable, os.path.join(COLAB_DIR, "evaluate.py"), "convert", export,
                                     ORIGINAL, cls.out, "--skip", "0"], capture_output=True, text=True, timeout=300)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_writes_every_condition_for_every_clip(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr[-2000:])
        clips = [r["clip"] for r in ev.read_tsv(os.path.join(self.out, "reference.tsv"))]
        self.assertEqual(len(clips), 2)
        for cond in ev.CONDITIONS:
            for clip in clips:
                info = sf.info(os.path.join(self.out, cond, clip + ".wav"))
                self.assertEqual(info.samplerate, ev.SR, cond)
                self.assertGreater(info.duration, 0.5, cond)

    def test_same_encoder_gives_same_distances(self):
        for row in ev.read_tsv(os.path.join(self.out, "units.tsv")):
            self.assertEqual(row["whisper_ft"], row["whisper_orig"])
            self.assertEqual(float(row["normal_ft"]), 0.0)

    def test_refuses_when_nothing_is_held_out(self):
        result = subprocess.run([sys.executable, os.path.join(COLAB_DIR, "evaluate.py"), "convert",
                                 os.path.join(self.tmp.name, "export"), ORIGINAL, os.path.join(self.tmp.name, "x"),
                                 "--skip", "2"], capture_output=True, text=True, timeout=300)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no held-out clips", result.stderr)


if __name__ == "__main__":
    unittest.main()
