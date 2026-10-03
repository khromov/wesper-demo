from __future__ import division

import argparse
import re
import sys
import os
import pyaudio
import numpy as np
import tkinter as tk
from tkinter import ttk
import time
import zmq
import soundfile as sf
import subprocess
import datetime
import torch
import sounddevice as sd


from direct import MyAudioClientDirect

# Audio recording parameters
RATE = 16000
FORMAT = pyaudio.paInt16
CHUNK=1024
CHANNELS=1
PORT=5557


class MyAudioClient(object):
    def __init__(self, host=None, port=PORT, sr=16000):
        assert host is not None
        server_address=f"tcp://{host}:{port}"
        print("### connectiong", server_address)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(server_address)
        self.save_root = "convlog"
        self.sr = sr

    def destroy(self):
        self.socket.close()
        self.context.destroy()

    def call(self, audio):
        #print("CALL ", len(audio))
        print("### SEND: ", type(audio), len(audio))
        start_time = time.time()
        self.socket.send_pyobj(audio)
        res = self.socket.recv_pyobj()  # [id, is_whisper, is_normal, rmx, trans_list]
        process_time = (time.time() - start_time)
        sec = (len(res[0]) / self.sr) 
        assert res is not None
        print("### RES", len(res), len(res[0]), sec, process_time, sec / process_time)
        # test playback
        sd.play(res[0], self.sr)
        sd.wait()

        '''
        # record converted samples
        now = datetime.datetime.now()
        fname = now.strftime('%y%m%d-%H%M%S')
        sf.write(os.path.join(self.save_root, f"{fname}-from.wav"), audio, 16000)
        sf.write(os.path.join(self.save_root, f"{fname}-to.wav"), res[0], 16000)        
        '''



class MicrophonePA(object):
    """ 押している間だけ録音するマイク """

    def __init__(self, rate=RATE, chunk=CHUNK):
        self._rate = rate
        self._chunk = chunk
        self._buff = []
        self.closed = True
        self._audio_stream = None
        self._audio_interface = pyaudio.PyAudio()

    def start_recording(self):
        self._buff = []
        assert self._audio_stream is None
        #assert self._audio_interface is None
        #self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        print("start recording")
        return self

    def stop_recording(self):
        assert not self.closed and self._audio_stream is not None
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self._audio_stream = None
        #self._audio_interface.terminate()
        #self._audio_interface = None
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        data = b''.join(self._buff)
        print("##### data len", len(data), type(data), type(data[0]))
        audio = np.frombuffer(data, dtype="int16") / 32768.0
        print("#### audio", type(audio), type(audio[0]), "len", len(audio), audio.max(), audio.min())
        self._buff = []
        print("stop recording")
        return audio

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.append(in_data)
        return None, pyaudio.paContinue

### mic  sounddevice version
import queue
import threading

q = None
goon = True
def callback(indata, frames, time, status):
    global q, goon
    """This is called (from a separate thread) for each audio block."""
    if status:
        print("***", status, file=sys.stderr)
    d = indata.copy()
    #print("#", type(d), len(d), d.dtype, max(d), min(d))
    q.put(d)

def mic_stream():
    global goon
    print("** enter thread loop")
    with sd.InputStream(samplerate=16000, channels=1, callback=callback):
        while goon:
            sd.sleep(10)
    print("** exit thread loop")

class MicrophoneSD(object):
    """ 押している間だけ録音するマイク """

    def __init__(self, rate=RATE, chunk=CHUNK):
        self._rate = rate
        self.recording = False

    def start_recording(self):
        global q, goon
        if self.recording:
            print("already recording")
            return

        self.recording = True
        q = queue.Queue()
        goon = True
        self.thread = threading.Thread(target=mic_stream)
        self.thread.start()

    def stop_recording(self):
        global q, goon
        goon = False
        self.thread.join() # wait for thread termination
        data = []
        while not q.empty():
            d = q.get()
            #data.append(d)
            data += [d]
        self.thread = None
        self.recording = False
        print("stop recording", len(data))
        if len(data) < 0:
            print("### empty audio")
            return None
        data = np.concatenate(data)
        audio = data[:,0]
        #print("*** data len", data.shape, len(data), type(data), type(data[0]), data.max(), data.min())
        #audio = np.frombuffer(data, dtype="int16") / 32768.0
        #audio = np.frombuffer(data, dtype="int16") 
        print("#### audio", type(audio), audio.shape, type(audio[0]), len(audio), audio.max(), audio.min())
        return audio

def device_choices(kind):
    """ (index, label) of the audio devices that can record (kind="input") or play (kind="output") """
    channels = "max_input_channels" if kind == "input" else "max_output_channels"
    return [(i, f"{i}: {d['name']}") for i, d in enumerate(sd.query_devices()) if d[channels] > 0]

def select_device(kind, index):
    """ Make device `index` the default input or output, used for all recording and playback.
        Raises, leaving the defaults alone, if the device can't do 16 kHz mono. """
    check = sd.check_input_settings if kind == "input" else sd.check_output_settings
    check(device=index, samplerate=RATE, channels=1)
    devices = list(sd.default.device)
    devices[0 if kind == "input" else 1] = index
    sd.default.device = devices

import tkinter.font
class MyGUI(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

        self.button = tk.Button(self, text="Push, Whisper, then Release", width=40, height=4, font=("Helvetica", 40))
        self.text = tk.Text(self, width=60, height=10)
        self.vsb = tk.Scrollbar(self, command=self.text.yview)
        self.text.configure(yscrollcommand=self.vsb.set)

        self.button.pack(side="top")
        self.make_device_selectors().pack(side="top", fill="x", padx=10, pady=10)
        self.vsb.pack(side="right", fill="y")
        self.text.pack(side="bottom", fill="x")

        self.button.bind("<ButtonPress>", self.on_press)
        self.button.bind("<ButtonRelease>", self.on_release)

        self.button.bind("<KeyPress>", self.on_keypress)
        self.button.bind("<KeyRelease>", self.on_keyrelease)

        self.mic = MicrophoneSD() # SD Sound Device version
        #self.mic = MicrophonePA() # PA PyAudio verison 
    
    def make_device_selectors(self):
        """ Microphone and output dropdowns. Devices plugged in later need a restart to show up. """
        frame = tk.Frame(self)
        frame.columnconfigure(1, weight=1)
        self.device_boxes = {}
        for row, (kind, title) in enumerate((("input", "Microphone"), ("output", "Output"))):
            choices = device_choices(kind)
            box = ttk.Combobox(frame, state="readonly", values=[label for _, label in choices])
            self.show_device(box, choices, sd.default.device[row])
            box.bind("<<ComboboxSelected>>",
                     lambda event, kind=kind, box=box, choices=choices: self.on_device(kind, box, choices))
            tk.Label(frame, text=title).grid(row=row, column=0, sticky="w", padx=(0, 10))
            box.grid(row=row, column=1, sticky="ew", pady=2)
            self.device_boxes[kind] = (box, choices)
        return frame

    def show_device(self, box, choices, index):
        for pos, (i, _) in enumerate(choices):
            if i == index:
                box.current(pos)

    def on_device(self, kind, box, choices):
        index, label = choices[box.current()]
        try:
            select_device(kind, index)
            self.log(f"{kind}: {label}")
        except Exception as e:
            self.log(f"can't use {label} for {kind}: {e}")
            self.show_device(box, choices, sd.default.device[0 if kind == "input" else 1])

    def set_client(self, host, args):
        if host == 'direct':
            self.client = MyAudioClientDirect(args)
            w2n = self.client.w2n
            level = "unchanged" if w2n.target_dbfs is None else f"normalized to {w2n.target_dbfs} dBFS speech level"
            self.log(f"encoder: {os.path.basename(args.hubert)}, input {level}")
            self.log(f"decoder: {os.path.basename(args.fastspeech2)}")
        else:
            self.client = MyAudioClient(host=host)

    def on_press(self, event):
        self.log("button was pressed")
        self.mic.start_recording()
        self.log("start")

    def on_keypress(self, event):
        self.log(f"keypress {event.keycode}")

    def on_keyrelease(self, event):
        self.log(f"keyrelease {event.keycode}")

    def on_release(self, event):
        self.log("button was released")
        audio = self.mic.stop_recording()
        if audio is not None:
            print(len(audio), audio.dtype, max(audio), min(audio))
            self.log("client-call")
            self.client.call(audio)
            self.log("client-get")

    def log(self, message):
        now = time.strftime("%I:%M:%S", time.localtime())
        self.text.insert("end", now + " " + message.strip() + "\n")
        self.text.see("end")

def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    #elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    #    return "mps" # M1 mac GPU
    else:
        return "cpu"
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--server",
        default = 'direct' # no server-client connection
    )

    ### for Direct Whisper-normal object
    parser.add_argument("--preprocess_config",
        default = 'config/my_preprocess16k_LJ.yaml'
    )

    parser.add_argument("--model_config",
        default = 'config/my_model16000.yaml'
    )

    parser.add_argument("--device", 
        default = get_default_device()
    )

    parser.add_argument("--port", type=int, default=PORT, help="server port number")

    parser.add_argument("--hubert", 
        help="hubert checkpoint path",
        #default="models/hubert/model-layer12-450000.pt"
        default="https://github.com/rkmt/wesper-demo/releases/download/v0.1/model-layer12-450000.pt",

    )

    parser.add_argument("--fastspeech2", 
        help="fastspeech2 checkpoint path",
        #default="models/fastspeech2/lambda_best.tar"
        #default="models/fastspeech2/googletts_neutral_best.tar"        
        default="https://github.com/rkmt/wesper-demo/releases/download/v0.1/googletts_neutral_best.tar",
    )

    parser.add_argument("--hifigan", 
        help="hifigan checkpoint path",
        #default="./hifigan/g_00205000"
        default="https://github.com/rkmt/wesper-demo/releases/download/v0.1/g_00205000",
    )


    parser.add_argument("--sd",
        help="input (microphone) device index, from the device list printed at startup; playback stays on the default output",
        type=int,
        default=-1,
    )


    args = parser.parse_args()   
    host = args.server
    print("### args", host, args)

    if args.sd >= 0:
        sd.default.device = (args.sd, sd.default.device[1])

    devices = sd.query_devices()
    print(devices)

    print(sd.default.device)

    root = tk.Tk()
    mygui = MyGUI(root)
    mygui.pack(side="top", fill="both", expand=True)

    mygui.set_client(host, args)
    root.mainloop()

