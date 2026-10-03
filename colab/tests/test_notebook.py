"""Tests for colab/wesper_sv_encoder_finetune.ipynb.

The quick tests check that the notebook's audio helpers agree with prepare_data.py, which
prepared the data the notebook trains on.

The smoke test runs every code cell on a tiny synthetic dataset (CPU, a few steps), then runs
it again to check that training resumes. It's opt-in, because the first run downloads WESPER's
checkpoints (about 1.5 GB, cached in torch's hub directory afterwards):

    WESPER_NOTEBOOK_SMOKE=1 .venv/bin/python -m unittest discover -s colab/tests -v

If the smoke test runs longer than WESPER_NOTEBOOK_SMOKE_TIMEOUT seconds (default 900), a
watchdog prints every thread's stack, showing where it hung, and exits.
"""
import ast
import contextlib
import faulthandler
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
import warnings

import numpy as np
import soundfile as sf

# Outside Jupyter, matplotlib on macOS opens a window and plt.show() blocks until it's closed.
os.environ.setdefault("MPLBACKEND", "Agg")

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
COLAB_DIR = os.path.dirname(TESTS_DIR)
REPO_DIR = os.path.dirname(COLAB_DIR)
NOTEBOOK = os.path.join(COLAB_DIR, "wesper_sv_encoder_finetune.ipynb")
sys.path.insert(0, COLAB_DIR)
sys.path.insert(0, TESTS_DIR)

import prepare_data as pd  # noqa: E402
from test_prepare_data import clip_with_thud, make_fake_corpus, noise, silence  # noqa: E402


def code_cells():
    with open(NOTEBOOK) as f:
        return [c["source"] for c in json.load(f)["cells"] if c["cell_type"] == "code"]


def notebook_function(name, namespace):
    """Compile one top-level function from the notebook into namespace and return it."""
    for source in code_cells():
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                exec(compile(ast.Module([node], type_ignores=[]), NOTEBOOK, "exec"), namespace)
                return namespace[name]
    raise LookupError(f"no function {name} in the notebook")


class MatchesPrepareData(unittest.TestCase):
    def setUp(self):
        self.ns = {"np": np, "sf": sf, "HOP": pd.HOP, "TARGET_DBFS": pd.TARGET_DBFS, "MAX_GAIN_DB": pd.MAX_GAIN_DB}

    def test_speech_dbfs_is_identical(self):
        speech_dbfs = notebook_function("speech_dbfs", self.ns)
        clips = [noise(1, -20), noise(0.5, -55), silence(1), np.zeros(5, np.float32), clip_with_thud()[0],
                 np.concatenate([silence(1), noise(2, -30), silence(0.3)])]
        for x in clips:
            self.assertEqual(speech_dbfs(x), pd.speech_dbfs(x))

    def test_normalize_matches_training_level(self):
        notebook_function("speech_dbfs", self.ns)
        normalize = notebook_function("normalize", self.ns)
        for level in (-45, -20, -5):
            self.assertAlmostEqual(pd.speech_dbfs(normalize(noise(1, level))), pd.TARGET_DBFS, delta=0.1)

    def test_normalize_caps_gain_like_prepare_data(self):
        notebook_function("speech_dbfs", self.ns)
        normalize = notebook_function("normalize", self.ns)
        self.assertAlmostEqual(pd.speech_dbfs(normalize(noise(1, -75))), -75 + pd.MAX_GAIN_DB, delta=0.5)

    def test_load_pair_undoes_storage_headroom(self):
        # prepare_data.py stores clips HEADROOM_DB down; load_pair must bring them back to TARGET_DBFS.
        with tempfile.TemporaryDirectory() as tmp:
            normal, whisper, _ = pd.process(noise(1, -33), noise(1, -50))
            for kind, x in (("normal", normal), ("whisper", whisper)):
                os.makedirs(os.path.join(tmp, kind))
                with open(os.path.join(tmp, kind, "c.flac"), "wb") as f:
                    f.write(pd.flac_bytes(x))
            self.ns.update(LOCAL_DATA_DIR=tmp, HEADROOM=10 ** (pd.HEADROOM_DB / 20))
            loaded = notebook_function("load_pair", self.ns)("c")
            for x in loaded:
                self.assertAlmostEqual(pd.speech_dbfs(x), pd.TARGET_DBFS, delta=0.3)
            self.assertEqual(len(loaded[0]), len(loaded[1]))

    def test_settings_come_from_prep_json(self):
        source = "\n".join(code_cells())
        self.assertIn('PREP["target_dbfs"]', source)
        self.assertIn('PREP["headroom_db"]', source)
        self.assertNotIn("TARGET_DBFS = -", source)  # no second, hard-coded copy


def run_notebook(overrides):
    """Execute every code cell, applying overrides right after the Settings cell. Returns (namespace, stdout)."""
    ns = {"__name__": "__main__"}
    out = io.StringIO()
    cwd = os.getcwd()
    try:
        with contextlib.redirect_stdout(out):
            for source in code_cells():
                exec(compile(source, NOTEBOOK, "exec"), ns)
                if "# @title Settings" in source:
                    ns.update(overrides)
    finally:
        os.chdir(cwd)  # the notebook changes into the WESPER repo
    return ns, out.getvalue()


@unittest.skipUnless(os.environ.get("WESPER_NOTEBOOK_SMOKE"), "set WESPER_NOTEBOOK_SMOKE=1 to run the notebook")
class Smoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Covers the whole class, including the resume run. Cancelled in tearDownClass.
        faulthandler.dump_traceback_later(int(os.environ.get("WESPER_NOTEBOOK_SMOKE_TIMEOUT", 900)), exit=True)
        try:
            import IPython.display  # noqa: F401
        except ImportError:  # the listen cells only need these to exist outside Jupyter
            display = types.ModuleType("IPython.display")
            display.Audio, display.display = (lambda *a, **k: None), (lambda *a, **k: None)
            ipython = types.ModuleType("IPython")
            ipython.get_ipython = lambda: None  # matplotlib probes this once IPython is importable
            ipython.display = display
            sys.modules["IPython"], sys.modules["IPython.display"] = ipython, display
        cls.tmp = tempfile.TemporaryDirectory()
        corpus, n2w = make_fake_corpus(cls.tmp.name)
        drive = os.path.join(cls.tmp.name, "drive")
        subprocess.run([sys.executable, os.path.join(COLAB_DIR, "prepare_data.py"), corpus, n2w,
                        os.path.join(drive, "data.tar"), "--val-clips", "2"], check=True, capture_output=True,
                       timeout=300)
        cls.overrides = dict(
            DRIVE_DIR=drive, DATA_TAR="data.tar", RUN_NAME="smoke", LOCAL_DATA_DIR=os.path.join(cls.tmp.name, "data"),
            WESPER_DIR=REPO_DIR, MAX_STEPS=4, BATCH_SIZE=2, CROP_SECONDS=1.0, WARMUP_STEPS=2,
            EVAL_EVERY=2, SAVE_EVERY=2, VAL_CLIPS=2, LISTEN_CLIPS=1, NUM_WORKERS=0,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Matplotlib is currently using agg")
            cls.ns, cls.out = run_notebook(cls.overrides)
        cls.run_dir = cls.ns["RUN_DIR"]
        # Read now: test_resumes_after_restart trains further in the same run folder.
        with open(os.path.join(cls.run_dir, "history.json")) as f:
            cls.history = json.load(f)

    @classmethod
    def tearDownClass(cls):
        faulthandler.cancel_dump_traceback_later()
        cls.tmp.cleanup()

    def test_trains_to_max_steps_and_saves(self):
        self.assertIn("Done at step 4", self.out)
        for name in ("latest.pt", "encoder_best.pt", "history.json"):
            self.assertTrue(os.path.exists(os.path.join(self.run_dir, name)), name)
        self.assertEqual([h["step"] for h in self.history["val"]], [0, 2, 4])

    def test_best_encoder_loads_with_wespers_own_loader(self):
        # convert.py --hubert uses whisper_normal.load_hubert, so the saved file must work there.
        import torch
        from whisper_normal import load_hubert
        encoder = load_hubert(os.path.join(self.run_dir, "encoder_best.pt"), device="cpu")
        units = encoder.units(torch.zeros(1, 1, 16000))
        self.assertEqual(tuple(units.shape), (1, 50, 256))

    def test_listen_cells_convert_audio(self):
        wav = self.ns["convert"](noise(1, -30), self.ns["finetuned"])
        self.assertGreater(len(wav), 8000)
        self.assertTrue(np.isfinite(wav).all())

    def test_resumes_after_restart(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Matplotlib is currently using agg")
            _, out = run_notebook({**self.overrides, "MAX_STEPS": 6})
        self.assertIn("Resuming from step 4", out)
        self.assertIn("Done at step 6", out)


if __name__ == "__main__":
    unittest.main()
