"""Tests for the GUI's microphone and output selectors in client_direct.py.

They use a fake device list, so they don't depend on what's plugged in. The GUI test builds
the real window but keeps it hidden.

    .venv/bin/python -m unittest discover -s tests -v
"""
import os
import sys
import tkinter as tk
import unittest
from unittest import mock

import sounddevice as sd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cwd = os.getcwd()
os.chdir(REPO)  # client_direct imports whisper_normal, which uses paths relative to the repo
sys.path.insert(0, REPO)
import client_direct as cd  # noqa: E402

FAKE_DEVICES = [
    {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2},
    {"name": "Interface", "max_input_channels": 2, "max_output_channels": 2},
    {"name": "Built-in Mic", "max_input_channels": 1, "max_output_channels": 0},
]


def tearDownModule():
    os.chdir(_cwd)


class DeviceSelection(unittest.TestCase):
    def setUp(self):
        self.saved = list(sd.default.device)
        sd.default.device = [2, 0]
        patches = [mock.patch.object(sd, "query_devices", return_value=FAKE_DEVICES),
                   mock.patch.object(sd, "check_input_settings"), mock.patch.object(sd, "check_output_settings")]
        self.query, self.check_in, self.check_out = (p.start() for p in patches)
        for p in patches:
            self.addCleanup(p.stop)

    def tearDown(self):
        sd.default.device = self.saved

    def test_lists_only_devices_that_can_record_or_play(self):
        self.assertEqual(cd.device_choices("input"), [(1, "1: Interface"), (2, "2: Built-in Mic")])
        self.assertEqual(cd.device_choices("output"), [(0, "0: Speakers"), (1, "1: Interface")])

    def test_selecting_input_keeps_output(self):
        cd.select_device("input", 1)
        self.check_in.assert_called_once_with(device=1, samplerate=16000, channels=1)
        self.assertEqual(list(sd.default.device), [1, 0])

    def test_selecting_output_keeps_input(self):
        cd.select_device("output", 1)
        self.check_out.assert_called_once_with(device=1, samplerate=16000, channels=1)
        self.assertEqual(list(sd.default.device), [2, 1])

    def test_unusable_device_raises_and_changes_nothing(self):
        self.check_in.side_effect = sd.PortAudioError("Invalid sample rate")
        with self.assertRaises(sd.PortAudioError):
            cd.select_device("input", 1)
        self.assertEqual(list(sd.default.device), [2, 0])


class Gui(DeviceSelection):
    def setUp(self):
        super().setUp()
        try:
            self.root = tk.Tk()
        except tk.TclError as e:
            self.skipTest(f"no display for Tk: {e}")
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.gui = cd.MyGUI(self.root)

    def choose(self, kind, label):
        box, choices = self.gui.device_boxes[kind]
        box.current([l for _, l in choices].index(label))
        self.gui.on_device(kind, box, choices)

    def log_text(self):
        return self.gui.text.get("1.0", "end")

    def test_dropdowns_show_current_devices(self):
        self.assertEqual(self.gui.device_boxes["input"][0].get(), "2: Built-in Mic")
        self.assertEqual(self.gui.device_boxes["output"][0].get(), "0: Speakers")
        self.assertEqual(list(self.gui.device_boxes["input"][0]["values"]), ["1: Interface", "2: Built-in Mic"])

    def test_choosing_a_device_makes_it_the_default(self):
        self.choose("input", "1: Interface")
        self.choose("output", "1: Interface")
        self.assertEqual(list(sd.default.device), [1, 1])
        self.assertIn("input: 1: Interface", self.log_text())

    def test_unusable_device_is_reported_and_dropdown_resets(self):
        self.check_out.side_effect = sd.PortAudioError("Invalid number of channels")
        self.choose("output", "1: Interface")
        self.assertEqual(list(sd.default.device), [2, 0])
        self.assertEqual(self.gui.device_boxes["output"][0].get(), "0: Speakers")
        self.assertIn("can't use 1: Interface for output", self.log_text())


if __name__ == "__main__":
    unittest.main()
