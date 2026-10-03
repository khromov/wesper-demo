"""Tests for startgui.sh.

Each test copies the launcher scripts into a temporary repo with a stub .venv/bin/python that
prints what it was called with, so no models are loaded and no window opens.

    .venv/bin/python -m unittest discover -s tests -v
"""
import os
import shutil
import subprocess
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENCODER = "colab/data/runs/n2w-finetune/encoder_best.pt"
STUB_PYTHON = '#!/bin/bash\necho "cwd=$PWD"\necho "args=$*"\n'


class StartGui(unittest.TestCase):
    def setUp(self):
        self.repo = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.repo)
        for name in ("startgui.sh", "client_direct_sv.sh"):
            shutil.copy(os.path.join(REPO, name), self.repo)

    def add_venv(self):
        os.makedirs(os.path.join(self.repo, ".venv/bin"))
        python = os.path.join(self.repo, ".venv/bin/python")
        with open(python, "w") as f:
            f.write(STUB_PYTHON)
        os.chmod(python, 0o755)

    def add_encoder(self):
        os.makedirs(os.path.dirname(os.path.join(self.repo, ENCODER)))
        open(os.path.join(self.repo, ENCODER), "w").close()

    def run_startgui(self, *args):
        # Run from elsewhere, with no `python` on PATH, as in a fresh shell on macOS.
        return subprocess.run(["/bin/bash", os.path.join(self.repo, "startgui.sh"), *args],
                              cwd=tempfile.gettempdir(), env={"PATH": "/usr/bin:/bin"},
                              capture_output=True, text=True, timeout=30)

    def test_runs_swedish_gui_with_venv_python_from_any_directory(self):
        self.add_venv()
        self.add_encoder()
        result = self.run_startgui("--sd", "4")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"cwd={self.repo}", result.stdout)
        self.assertIn(f"args=client_direct.py --hubert {ENCODER} --sd 4", result.stdout)

    def test_missing_venv_is_reported(self):
        self.add_encoder()
        result = self.run_startgui()
        self.assertEqual(result.returncode, 1)
        self.assertIn("No .venv", result.stderr)

    def test_missing_encoder_is_reported(self):
        self.add_venv()
        result = self.run_startgui()
        self.assertEqual(result.returncode, 1)
        self.assertIn("Swedish encoder not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
