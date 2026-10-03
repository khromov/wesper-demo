"""Keeps the browser demo's port of speech_dbfs() (web/src/lib/audio/level.ts) in step with Python.

web/src/lib/audio/level.fixture.json holds speech levels and normalization gains that
colab/prepare_data.py computes for a few deterministic signals; web's `bun test` checks the
TypeScript port against the same numbers. This test fails if the Python side no longer
produces them. After a deliberate change to speech_dbfs(), regenerate the fixture with:

    WESPER_WRITE_FIXTURE=1 .venv/bin/python -m unittest tests.test_web_level_fixture
"""
import json
import math
import os
import sys
import unittest

import numpy as np
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO, "web", "src", "lib", "audio", "level.fixture.json")
sys.path.insert(0, REPO)
from colab.prepare_data import speech_dbfs  # noqa: E402

SR = 16000
TARGET_DBFS, MAX_GAIN_DB = -20.0, 40.0  # the Swedish encoder's settings


def lcg(seed, n):
    """Uniform noise in [-1, 1) from a 32-bit LCG; level.test.ts generates the same numbers."""
    out, state = np.empty(n), seed
    for i in range(n):
        state = (1664525 * state + 1013904223) % 2 ** 32
        out[i] = state / 2 ** 32 * 2 - 1
    return out


def tone(freq, amp, n):
    return amp * np.sin(2 * math.pi * freq * np.arange(n) / SR)


def signals():
    """name -> float32 signal. Must match signals() in level.test.ts."""
    t = np.arange(2 * SR) / SR
    click = tone(330, 0.05, int(1.5 * SR))
    click[8000] = 0.9
    whisper, sr = sf.read(os.path.join(REPO, "sample_whisper.wav"), dtype="float32")
    assert sr == SR
    s = {
        "silence": np.zeros(SR),
        "short": tone(220, 0.1, 100),
        "tone_padded": np.concatenate([np.zeros(SR // 2), tone(220, 0.1, SR), np.zeros(SR // 2)]),
        "noise": 0.01 * lcg(1, SR),
        "tone_click": click,
        "modulated": 0.2 * np.sin(2 * math.pi * 150 * t) * (0.5 + 0.5 * np.sin(2 * math.pi * 3 * t)) + 1e-4 * lcg(7, 2 * SR),
        "quiet_noise": 1e-5 * lcg(3, SR),
    }
    s = {k: v.astype(np.float32) for k, v in s.items()}
    s["sample_whisper"] = whisper
    return s


def expected():
    out = {}
    for name, x in signals().items():
        level = speech_dbfs(x)
        out[name] = {"speechDbfs": level, "gainDb": min(TARGET_DBFS - level, MAX_GAIN_DB)}
    return {"targetDbfs": TARGET_DBFS, "maxGainDb": MAX_GAIN_DB, "signals": out}


class LevelFixture(unittest.TestCase):
    def test_fixture_matches_python(self):
        exp = expected()
        if os.environ.get("WESPER_WRITE_FIXTURE"):
            with open(FIXTURE, "w") as f:
                json.dump(exp, f, indent=2)
                f.write("\n")
        with open(FIXTURE) as f:
            fixture = json.load(f)
        self.assertEqual(fixture.keys(), exp.keys())
        self.assertEqual(fixture["signals"].keys(), exp["signals"].keys())
        for name, values in exp["signals"].items():
            for key, value in values.items():
                self.assertAlmostEqual(fixture["signals"][name][key], value, places=9, msg=f"{name}.{key}")

    def test_signals_cover_the_interesting_cases(self):
        levels = {k: v["speechDbfs"] for k, v in expected()["signals"].items()}
        self.assertEqual(levels["silence"], -120.0)
        self.assertEqual(levels["short"], -120.0)  # shorter than one 20 ms frame
        self.assertGreater(levels["quiet_noise"], -120.0)
        self.assertLess(levels["quiet_noise"] + 40, TARGET_DBFS)  # the gain cap applies


if __name__ == "__main__":
    unittest.main()
