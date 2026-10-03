import sys
sys.path.append("libs/FastSpeech2")

import numpy as np
import librosa
import soundfile as sf
import json
import torch
import yaml
import argparse
import os

import torch.nn.functional as F
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

# FastSpeech2
from model import FastSpeech2
from utils.model import get_vocoder
import utils.tools
print("##### utils.tools.device", utils.tools.device)
import model.modules
print("#### model.moculde.device", model.modules.device)
from utils.tools import pad_2D
from utils.model import vocoder_infer
# HuBERT
from libs.hubert.model import Hubert, URLS, HubertSoft
# Hifigan
import hifigan
# The speech-level measure the fine-tuned encoders' training data was normalized with
from colab.prepare_data import speech_dbfs
# The vocoders a decoder can be trained for (HiFi-GAN 16 kHz, BigVGAN)
import vocoders


def load_fastspeech2(configs, checkpoint_path=None, device='cuda'):
    assert checkpoint_path is not None
    (preprocess_config, model_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    print("### loading FastSpeech2", checkpoint_path)
    if checkpoint_path.startswith("http"):
        ckpt = torch.hub.load_state_dict_from_url(checkpoint_path, map_location=torch.device('cpu')) if device!='cuda' else torch.hub.load_state_dict_from_url(checkpoint_path)
    else:
        ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu')) if device!='cuda' else torch.load(checkpoint_path)
    model.load_state_dict(ckpt["model"], strict=True)

    model = model.to(device)
    model.eval()
    model.requires_grad_ = False
    return model

def units2wav(units, fs2model, vocoder, model_config, preprocess_config, device='cuda', d_targets=None, p_targets=None, e_targets=None, vocoder_spec=None):
    ''' units: [N, D=256] 
        fs2model: Unit-FastSpeech2
        vocoder: HiFi-GAN, or the vocoder described by vocoder_spec (vocoders.py)
    '''
    print("### units2wav", "pitch", p_targets) 
    if type(units) == torch.Tensor:
        if len(units.shape) == 3:
            units = units.squeeze(0)
        units = units.detach().cpu().numpy()
        assert len(units.shape) == 2
    if vocoder_spec is not None and not vocoder_spec.one_frame_per_unit:
        # The decoder works at the vocoder's frame rate: units interpolated to its frames, as in training.
        units = vocoders.units_to_frames(units, vocoders.n_frames(len(units), vocoder_spec), vocoder_spec)
    units = pad_2D([units])
    #units = torch.from_numpy(units).long().to(device)
    units = torch.from_numpy(units).to(device)
    speakers = torch.tensor([0], device=device) #batch_t[2]
    max_src_len = units.shape[1]
    src_lens = torch.tensor([max_src_len], device=device) #batch_t[4]

    with torch.no_grad(): # predict MELs from UNITs by FastSpeech2
        #print("## device", speakers.device, units.device, src_lens.device)
        # added rkmt 2023.7.20
        if d_targets is None:
            d_targets = torch.tensor([1]*(units.shape[1]), device=device).unsqueeze(0) # use fixed duration
        if p_targets is not None:
            p_targets = np.array(p_targets) if type(p_targets) == list else p_targets
            p_targets = torch.tensor(p_targets, dtype=torch.float32).unsqueeze(0).to(device)
            print("p_targets", p_targets.shape)
        if e_targets is not None:
            e_targets = np.array(e_targets) if type(e_targets) == list else e_targets
            e_targets = torch.tensor(e_targets, dtype=torch.float32).unsqueeze(0).to(device)
            print("e_targets", e_targets.shape)
        print("### units", units.shape, "d_targets", d_targets.shape)
        output = fs2model(speakers, units, src_lens, max_src_len) #d_targets=d_targets, p_targets=p_targets, e_targets=e_targets
    mel_len = output[9][0].item() # mel_lens
    mel_prediction = output[1][0, :mel_len].detach().transpose(0, 1) # postnet_output
    
    if vocoder_spec is not None and vocoder_spec.name != "hifigan16k":
        # 16-bit like vocoder_infer's output, but clipped instead of wrapping around.
        wav = vocoders.synthesize(vocoder, mel_prediction)
        return (np.clip(wav, -1, 1) * 32767).astype("int16"), output
    with torch.no_grad(): # predict Wavs from MELs
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    return wav_prediction, output

def wav2units(wav, encoder, layer=None, device='cuda'):
    ''' 
        encoder: HuBERT
    '''
    if type(wav) == np.ndarray:
        wav = torch.tensor([wav], dtype=torch.float32, device=device)
    else:
        wav = wav.to(device)
    assert type(wav) == torch.Tensor
    if len(wav.shape) == 2:
        wav = wav.unsqueeze(0)
    #print("#wav2units: ", wav.dtype, wav.shape, min(wav), max(wav))
    with torch.no_grad():  # wav -> HuBERT soft units
        if layer is None or layer < 0:
            #print("#encoder", type(encoder), "device", (next(encoder.parameters())).device)
            #print("#WAV", type(wav), wav.device)
            units = encoder.units(wav) 
        else:
            wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
            units, _ = encoder.encode(wav, layer=layer)
            
    #print("Units", units.shape)
    
    return units


def load_hubert(checkpoint_path=None, rank=0, device='cuda'):
    print("### load_hubert", checkpoint_path, device)
    assert checkpoint_path is not None
    print("### loading checkpoint from: ", checkpoint_path)
    if checkpoint_path.startswith("http"):
        checkpoint = torch.hub.load_state_dict_from_url(checkpoint_path, map_location=torch.device('cpu')) if device!='cuda' else torch.hub.load_state_dict_from_url(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu')) if device!='cuda' else torch.load(checkpoint_path)
    hubert = HubertSoft().to(device) if device!='cuda' else HubertSoft().to(rank)

    # Encoders fine-tuned with colab/wesper_sv_encoder_finetune.ipynb also store their training
    # settings, such as the speech level their input was normalized to. The original has none.
    training_config = checkpoint.get('config') or {}
    checkpoint = checkpoint['hubert'] if checkpoint['hubert'] is not None else checkpoint
    consume_prefix_in_state_dict_if_present(checkpoint, "module.")

    hubert.load_state_dict(checkpoint, strict=True)
    hubert.eval().to(device)
    hubert.training_config = training_config
    return hubert


def normalize_level(wav, target_dbfs, max_gain_db=40.0):
    ''' Scale audio to target_dbfs speech level, amplifying by at most max_gain_db.
        Keeps the input's shape; the result stays float, so peaks above 1.0 are kept, not clipped.
    '''
    if type(wav) == torch.Tensor:
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32)
    gain_db = min(target_dbfs - speech_dbfs(wav.reshape(-1)), max_gain_db)
    return wav * np.float32(10 ** (gain_db / 20))


def load_hifigan(config, checkpoint_path="./hifigan/g_00205000", device='cuda'):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]
    assert speaker == 'universal'
    assert name == "HiFi-GAN16k"

    print("#### HiFI-GAN16k", name, speaker, device)
    with open("./hifigan/my_config_v1_16000.json", "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    print("### HiFI-GAN ckpt", checkpoint_path)
    if checkpoint_path.startswith("http"):
        ckpt = torch.hub.load_state_dict_from_url(checkpoint_path, map_location=torch.device('cpu')) if device!='cuda' else torch.hub.load_state_dict_from_url(checkpoint_path)
    else:
        ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu')) if device!='cuda' else torch.load(checkpoint_path)

    vocoder.load_state_dict(ckpt['generator'])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)

    return vocoder



class MyWhisper2Normal(object):
    def __init__(self, args, load_encoder=True, load_decoder=True, load_vocoder=True, root=None):
        if root is None:
            print("## Lib Local Path", __file__)
            root = os.path.dirname(__file__)
        print("root", root)
        self.root = root

        self.device = device = args.device
        # set FastSpeech direct defined device
        utils.tools.device = device
        model.modules.device = device

        self.hubert = args.hubert # HuBert
        self.fastspeech2 = args.fastspeech2 # HuBert
        self.hifigan = args.hifigan
        self.args = args

        print("MyWhisper2Normal:args", args)

        # Read Config
        self.preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
        self.model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
        self.configs = (self.preprocess_config, self.model_config)
        
        # Input level. An encoder trained on loudness-normalized audio records the level in its
        # checkpoint, and its input (e.g. the microphone) is normalized the same way. WESPER's
        # original encoder records none, so its input is passed through unchanged.
        self.target_dbfs, self.max_gain_db = None, None
        if load_encoder:
            print("#### loading HuBERT")
            self.encoder = load_hubert(args.hubert, device=device) # device
            print("### HuBERT model", type(self.encoder))
            self.target_dbfs = self.encoder.training_config.get("TARGET_DBFS")
            self.max_gain_db = self.encoder.training_config.get("MAX_GAIN_DB", 40.0)
            if self.target_dbfs is not None:
                print(f"### normalizing input to {self.target_dbfs} dBFS speech level, as the encoder was trained")
        
        # FastSpeech2 HiFi-GAN
        if load_decoder:
            print("#### loading FastSpeech2")
            self.fs2model = load_fastspeech2(self.configs, checkpoint_path=args.fastspeech2, device=device) # load FastSpeech2

        # The vocoder the decoder was trained for, named in its preprocess config (decoder/train.py).
        # Decoders without that entry, like WESPER's released ones, use WESPER's HiFi-GAN.
        self.vocoder_spec = vocoders.spec(self.preprocess_config.get("vocoder", {}).get("name", vocoders.DEFAULT))
        self.sample_rate = self.vocoder_spec.sample_rate  # of the converted audio
        if load_vocoder:
            if self.vocoder_spec.name == "hifigan16k":
                print("### loading HiFI GAN")
                self.vocoder = load_hifigan(self.model_config, checkpoint_path=args.hifigan, device=device).eval()
            else:
                print("### loading", self.vocoder_spec.name)
                self.vocoder = vocoders.load(self.vocoder_spec, device)
        print("#### Done.")

    def test(self, wavfile='sample_whisper.wav', outfile='/tmp/out.wav'):
        print("### Convesion test")
        wav, sr = librosa.load(wavfile, sr=16000)
        wav_to, mel = self.convert(wav)
        print("### test:converted", type(wav_to), wav_to.shape)
        #wav_to = torch.tensor(wav_to).unsqueeze(0)
        sf.write(outfile, wav_to, self.sample_rate)
        print("### test saved to ", outfile)
        return wav_to, mel


    def whisper2normal(self, wav_from):
        if self.target_dbfs is not None:
            wav_from = normalize_level(wav_from, self.target_dbfs, self.max_gain_db)
        if type(wav_from) != torch.Tensor:
            #wav_t = torch.tensor([wav_from], dtype=torch.float32).to(self.device)  # [1, LEN]
            wav_t = torch.tensor(wav_from, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, LEN]
        else:
            wav_t = wav_from.to(self.device)
        #print("#w2normal WAV", self.device, type(wav_t), wav_t.shape, "maxmin", wav_t.max(), wav_t.min())
        units = wav2units(wav_t, self.encoder, device=self.device) # self.device // cpu
        wav_to, _ = units2wav(units, self.fs2model, self.vocoder, self.model_config, self.preprocess_config, device=self.device, vocoder_spec=self.vocoder_spec)
        #print("#w2normal UNITS", units.shape, "WAV_TO", wav_to.dtype, wav_to.shape, wav_to.min(), wav_to.max())
        return wav_to, None
    
    def wav2units(self, wav_t):
        return wav2units(wav_t, self.encoder, device=self.device)
    
    def units2wav(self, units, p_targets=None, e_targets=None):
        return units2wav(units, self.fs2model, self.vocoder, self.model_config, self.preprocess_config, p_targets=p_targets, e_targets=e_targets, device=self.device, vocoder_spec=self.vocoder_spec)

    def convert(self, wav_from):
        wav_to, mel = self.whisper2normal(wav_from)
        return wav_to, mel

