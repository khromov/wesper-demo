"""Tests for the GUI's microphone, output and voice selectors in client_direct.py.

They use a fake device list, so they don't depend on what's plugged in. The GUI test builds
the real window but keeps it hidden.

    .venv/bin/python -m unittest discover -s tests -v
"""
import argparse
import os
import sys
import tempfile
import threading
import time
import tkinter as tk
import types
import unittest
from unittest import mock

import sounddevice as sd
import yaml

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


def make_runs(root):
    """A decoder/runs folder: runs for each vocoder, one without a decoder, and a stray file."""
    runs = os.path.join(root, "runs")
    for name, vocoder in (("a-hifi", {"name": "hifigan16k"}), ("b-big", {"name": "bigvgan22k"}),
                          ("c-vft", {"name": "bigvgan22k", "checkpoint": "bigvgan_generator.pt"}), ("d-old", None),
                          ("e-unfinished", {"name": "bigvgan22k"})):
        os.makedirs(os.path.join(runs, name))
        with open(os.path.join(runs, name, "preprocess.yaml"), "w") as f:
            yaml.safe_dump({"path": {"preprocessed_path": name}, **({"vocoder": vocoder} if vocoder else {})}, f)
        if name != "e-unfinished":
            open(os.path.join(runs, name, "decoder_best.pt"), "w").close()
    open(os.path.join(runs, "a-hifi.log"), "w").close()
    return runs


class Voices(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runs = make_runs(self.tmp.name)

    def test_english_then_each_finished_run_with_its_vocoder(self):
        voices = cd.voice_choices(self.runs)
        self.assertEqual([label for label, _, _ in voices], [
            "English (WESPER, HiFi-GAN 16 kHz)", "a-hifi (HiFi-GAN 16 kHz)", "b-big (BigVGAN 22 kHz)",
            "c-vft (BigVGAN 22 kHz, fine-tuned vocoder)", "d-old (HiFi-GAN 16 kHz)"])
        self.assertEqual(voices[0][1:], (cd.ENGLISH_DECODER, cd.ENGLISH_CONFIG))
        self.assertEqual(voices[3][1:], (os.path.join(self.runs, "c-vft", "decoder_best.pt"),
                                         os.path.join(self.runs, "c-vft", "preprocess.yaml")))

    def test_no_runs_folder_leaves_english(self):
        self.assertEqual(len(cd.voice_choices(os.path.join(self.tmp.name, "nowhere"))), 1)

    def test_same_files_compares_paths_not_spellings(self):
        a = (os.path.join(self.runs, "b-big", "decoder_best.pt"), os.path.join(self.runs, "b-big", "preprocess.yaml"))
        b = tuple(os.path.relpath(p) for p in a)
        self.assertTrue(cd.same_files(a, b))
        self.assertFalse(cd.same_files(a, (cd.ENGLISH_DECODER, cd.ENGLISH_CONFIG)))

    def test_load_voice_shares_the_encoder_and_level(self):
        loaded = []

        class FakeW2N:
            def __init__(self, args, load_encoder=True):
                loaded.append((args, load_encoder))
        old = types.SimpleNamespace(encoder="the encoder", target_dbfs=-20.0, max_gain_db=40.0)
        args = argparse.Namespace(hubert="enc.pt", fastspeech2="x/decoder_best.pt", preprocess_config="x/preprocess.yaml", device="cpu")
        with mock.patch.object(cd, "MyWhisper2Normal", FakeW2N):
            voice = cd.load_voice(old, args, "y/decoder_best.pt", "y/preprocess.yaml")
        (new_args, load_encoder), = loaded
        self.assertFalse(load_encoder)
        self.assertEqual((new_args.fastspeech2, new_args.preprocess_config, new_args.hubert), ("y/decoder_best.pt", "y/preprocess.yaml", "enc.pt"))
        self.assertEqual(args.fastspeech2, "x/decoder_best.pt")  # the caller's args are left alone
        self.assertEqual((voice.encoder, voice.target_dbfs, voice.max_gain_db), ("the encoder", -20.0, 40.0))


def fake_w2n(vocoder="bigvgan22k", checkpoint=None):
    rate = 16000 if vocoder == "hifigan16k" else 22050
    return types.SimpleNamespace(vocoder_spec=types.SimpleNamespace(name=vocoder), sample_rate=rate,
                                 vocoder_checkpoint=checkpoint, encoder="the encoder", target_dbfs=-20.0, max_gain_db=40.0)


class VoiceGui(DeviceSelection):
    def setUp(self):
        super().setUp()
        try:
            self.root = tk.Tk()
        except tk.TclError as e:
            self.skipTest(f"no display for Tk: {e}")
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runs = make_runs(self.tmp.name)
        self.gui = cd.MyGUI(self.root)
        self.gui.client = types.SimpleNamespace(w2n=fake_w2n())
        self.args = argparse.Namespace(fastspeech2=os.path.join(self.runs, "b-big", "decoder_best.pt"),
                                       preprocess_config=os.path.join(self.runs, "b-big", "preprocess.yaml"), voices=self.runs)
        self.load = mock.patch.object(cd, "load_voice", side_effect=lambda w2n, args, d, c: fake_w2n(checkpoint="vft.pt")).start()
        self.addCleanup(mock.patch.stopall)

    def choose(self, label):
        self.gui.voice_box.current(list(self.gui.voice_box["values"]).index(label))
        self.gui.on_voice()
        deadline = time.time() + 5
        while self.gui.loading and time.time() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertFalse(self.gui.loading)

    def log_text(self):
        return self.gui.text.get("1.0", "end")

    def test_dropdown_shows_the_current_voice(self):
        self.gui.make_voice_selector(self.args)
        self.assertEqual(self.gui.voice_box.get(), "b-big (BigVGAN 22 kHz)")
        self.assertEqual(len(self.gui.voice_box["values"]), 5)
        self.assertIn("voice: b-big (BigVGAN 22 kHz)", self.log_text())

    def test_choosing_a_voice_loads_it_in_the_background(self):
        self.gui.make_voice_selector(self.args)
        before = self.gui.client.w2n
        self.choose("c-vft (BigVGAN 22 kHz, fine-tuned vocoder)")
        w2n, args, decoder, config = self.load.call_args.args
        self.assertIs(w2n, before)
        self.assertEqual((decoder, config), (os.path.join(self.runs, "c-vft", "decoder_best.pt"),
                                             os.path.join(self.runs, "c-vft", "preprocess.yaml")))
        self.assertEqual(self.gui.client.w2n.vocoder_checkpoint, "vft.pt")
        self.assertIn("loading voice: c-vft", self.log_text())
        self.assertIn("vocoder: bigvgan22k (22050 Hz), fine-tuned: vft.pt", self.log_text())

    def test_switching_back_reuses_a_loaded_voice(self):
        self.gui.make_voice_selector(self.args)
        before = self.gui.client.w2n
        self.choose("c-vft (BigVGAN 22 kHz, fine-tuned vocoder)")
        self.choose("b-big (BigVGAN 22 kHz)")
        self.assertIs(self.gui.client.w2n, before)
        self.choose("c-vft (BigVGAN 22 kHz, fine-tuned vocoder)")
        self.assertEqual(self.load.call_count, 1)

    def test_button_does_nothing_while_loading(self):
        self.gui.make_voice_selector(self.args)
        release = threading.Event()
        self.load.side_effect = lambda *a: release.wait(5) and fake_w2n()
        self.gui.mic = mock.Mock(recording=False)
        self.gui.voice_box.current(1)
        self.gui.on_voice()
        self.gui.on_press(None)
        self.gui.on_release(None)
        self.gui.mic.start_recording.assert_not_called()
        self.gui.mic.stop_recording.assert_not_called()
        self.assertIn("still loading", self.log_text())
        release.set()
        self.choose(self.gui.voice_box.get())

    def test_a_voice_that_fails_to_load_is_reported_and_dropdown_resets(self):
        self.gui.make_voice_selector(self.args)
        before = self.gui.client.w2n
        self.load.side_effect = RuntimeError("no such checkpoint")
        self.choose("a-hifi (HiFi-GAN 16 kHz)")
        self.assertIs(self.gui.client.w2n, before)
        self.assertEqual(self.gui.voice_box.get(), "b-big (BigVGAN 22 kHz)")
        self.assertIn("can't load a-hifi (HiFi-GAN 16 kHz): no such checkpoint", self.log_text())

    def test_a_decoder_from_elsewhere_is_listed_too(self):
        other = os.path.join(self.tmp.name, "elsewhere")
        self.args.fastspeech2, self.args.preprocess_config = os.path.join(other, "decoder_best.pt"), os.path.join(other, "preprocess.yaml")
        self.gui.make_voice_selector(self.args)
        self.assertEqual(self.gui.voice_box.get(), other)
        self.assertEqual(len(self.gui.voice_box["values"]), 6)


if __name__ == "__main__":
    unittest.main()
