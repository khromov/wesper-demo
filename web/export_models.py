"""Export WESPER's models to ONNX for the browser demo in web/.

Writes into web/public/models/ (by default):
  encoder-<id>.onnx         audio [1, T] float32, 16 kHz -> soft units [1, T // 320, 256]
  decoder-<id>.onnx         soft units -> audio [1, S] float32 (FastSpeech2 + its vocoder), one per voice
  models.json               what the app loads: labels, files, sizes, hashes and each encoder's input level

The models stay fp32: fp16 versions were tried and sounded clearly worse in the browser.

Encoders: WESPER's original, and the Swedish fine-tuned one if its checkpoint exists. Decoders
(voices): the Swedish narrator trained by decoder/train.py, for HiFi-GAN and for BigVGAN, if their
run folders exist, and WESPER's English googletts one. A voice plays at its vocoder's sample rate
(16 kHz for HiFi-GAN, 22.05 kHz for BigVGAN-22k; in models.json). Every exported file is checked
against the PyTorch code path that convert.py and the GUI use; the script fails if they disagree.

    .venv/bin/python web/export_models.py               # needs: pip install -r web/requirements-export.txt

The wrappers below change how a few operations are written, never what they compute:
- nn.TransformerEncoderLayer's attention is spelled out, because the traced nn.MultiheadAttention
  bakes the sequence length into a Reshape (and its fused fast path has no ONNX export at all).
- torch.bucketize becomes a compare-and-count, and FastSpeech2's length regulator (a Python
  loop over .item() values) becomes a cumsum lookup.
- The decoder's position table is precomputed for 120 s; the original computes the same table
  on the fly beyond 1000 frames (20 s).
- For vocoders with other frame rates than the units (BigVGAN), the units are interpolated to the
  vocoder's frames inside the graph, as vocoders.units_to_frames does, with the frame count in
  the same integer arithmetic as vocoders.n_frames so it follows the input's length.
- BigVGAN's anti-aliasing filters are expanded to their input's channel count at run time, which
  the exporter can't size ("convolution for kernel of unknown shape"); each gets a fixed,
  pre-expanded kernel instead (fixed_bigvgan_filters).
"""
import argparse
import faulthandler
import hashlib
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "libs", "FastSpeech2"))
import vocoders  # noqa: E402
import whisper_normal as wn  # noqa: E402
from transformer.Models import get_sinusoid_encoding_table  # noqa: E402

RELEASE = "https://github.com/rkmt/wesper-demo/releases/download/v0.1"
SR, HOP = 16000, 320
MAX_SECONDS = 120  # longest input the decoder's position table covers
OPSET = 17


# ---------------------------------------------------------------- export wrappers

class ExportableEncoderLayer(nn.Module):
    """nn.TransformerEncoderLayer (post-norm, gelu, eval mode, batch 1, no masks), written out so
    that the traced graph keeps the sequence length dynamic. Wraps the layer without changing it."""

    def __init__(self, layer, heads):
        super().__init__()
        self.layer, self.heads = layer, heads

    def forward(self, x):
        l, H = self.layer, self.heads
        C = x.shape[-1]
        D = C // H
        q, k, v = F.linear(x, l.self_attn.in_proj_weight, l.self_attn.in_proj_bias).chunk(3, -1)
        q, k, v = (t.reshape(1, -1, H, D).transpose(1, 2) for t in (q, k, v))
        a = torch.softmax(q @ k.transpose(-1, -2) / D ** 0.5, -1) @ v
        x = l.norm1(x + l.self_attn.out_proj(a.transpose(1, 2).reshape(1, -1, C)))
        return l.norm2(x + l.linear2(F.gelu(l.linear1(x))))


class EncoderExport(nn.Module):
    """wav [1, T] -> soft units [1, T // 320, 256]: HubertSoft.units(), batch 1."""

    def __init__(self, hubert):
        super().__init__()
        self.h = hubert
        self.layers = nn.ModuleList(ExportableEncoderLayer(l, l.self_attn.num_heads) for l in hubert.encoder.layers)

    def forward(self, wav):
        h = self.h
        x = F.pad(wav[:, None, :], ((400 - 320) // 2, (400 - 320) // 2))
        x = h.feature_projection(h.feature_extractor(x).transpose(1, 2))
        x = h.norm(x + h.positional_embedding(x))
        for layer in self.layers:
            x = layer(x)
        return h.proj(x)


def bucketize(x, bins):
    """torch.bucketize(x, bins) (right=False): the number of bins strictly below x."""
    return (x.unsqueeze(-1) > bins).sum(-1)


def regulate_length(x, durations):
    """FastSpeech2's LengthRegulator for batch 1: repeat x[:, i] durations[i] times.
    Output frame t takes unit i where cumsum(d)[i - 1] <= t < cumsum(d)[i]."""
    ends = torch.cumsum(durations, 0)
    t = torch.arange(ends[-1], device=x.device)
    return x[:, (t.unsqueeze(1) >= ends.unsqueeze(0)).sum(1)]


def units_to_frames(units, voc):
    """vocoders.units_to_frames(units, vocoders.n_frames(N, voc), voc) for batch 1, written in tensor
    ops so the number of frames follows the input's length in the exported graph. Frame j's
    position among the units, (j + 0.5) * frame_seconds / UNIT_SECONDS - 0.5, is kept as an exact
    integer fraction (float32 positions would be off by 1e-4 after a minute)."""
    n_units = torch.ones_like(units[0, :, 0], dtype=torch.long).sum()
    n = n_units * (vocoders.UNIT_HOP * voc.sample_rate) // (vocoders.UNIT_SAMPLE_RATE * voc.hop)
    # position = num / den - 1, with num > 0 for every frame
    den = 2 * voc.sample_rate * vocoders.UNIT_HOP
    num = (2 * torch.arange(n, device=units.device) + 1) * (voc.hop * vocoders.UNIT_SAMPLE_RATE) + den // 2
    lo = num // den - 1
    w = (num % den).float() / den * (lo >= 0).float()  # before the first unit's center: the first unit
    lo = torch.minimum(lo.clamp(min=0), n_units - 1)  # after the last one's: the last unit
    hi = torch.minimum(lo + 1, n_units - 1)
    w = w[None, :, None]
    return units[:, lo] * (1 - w) + units[:, hi] * w


def _upsample_forward(self, x):
    """UpSample1d.forward with the pre-expanded filter (alias_free_activation/torch/resample.py)."""
    x = F.pad(x, (self.pad, self.pad), mode="replicate")
    x = self.ratio * F.conv_transpose1d(x, self.filter_c, stride=self.stride, groups=self.filter_c.shape[0])
    return x[..., self.pad_left: -self.pad_right]


def _lowpass_forward(self, x):
    """LowPassFilter1d.forward with the pre-expanded filter (alias_free_activation/torch/filter.py)."""
    if self.padding:
        x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
    return F.conv1d(x, self.filter_c, stride=self.stride, groups=self.filter_c.shape[0])


def fixed_bigvgan_filters(vocoder, n_mels):
    """Gives each of BigVGAN's anti-aliasing filters a fixed kernel, expanded to the channel count it
    sees (recorded with one forward pass), in place of expanding it at run time. Same output.
    Returns how many filters were changed."""
    from libs.bigvgan.alias_free_activation.torch.filter import LowPassFilter1d
    from libs.bigvgan.alias_free_activation.torch.resample import UpSample1d
    channels = {}
    hooks = [m.register_forward_pre_hook(lambda m, inp: channels.__setitem__(m, inp[0].shape[1]))
             for m in vocoder.modules() if isinstance(m, (UpSample1d, LowPassFilter1d))]
    vocoder(torch.zeros(1, n_mels, 16))
    for h in hooks:
        h.remove()
    for m, c in channels.items():
        m.register_buffer("filter_c", m.filter.expand(c, -1, -1).contiguous())
        m.forward = types.MethodType(_upsample_forward if isinstance(m, UpSample1d) else _lowpass_forward, m)
    return len(channels)


class DecoderExport(nn.Module):
    """soft units [1, N, 256] -> wav [1, S]: FastSpeech2's inference path as units2wav() runs it
    (no targets: predicted pitch, energy and durations), then the vocoder (voc). Batch 1, no
    padding. For vocoders with other frame rates, the units are first moved to its frames."""

    def __init__(self, fs2, vocoder, voc=vocoders.HIFIGAN16K):
        super().__init__()
        assert isinstance(fs2.encoder.src_word_emb, nn.Identity), "expects 256-dim soft units (soft_unit_dim 256)"
        self.fs2, self.vocoder, self.voc = fs2, vocoder, voc
        d = fs2.encoder.d_model
        frames = vocoders.n_frames(MAX_SECONDS * SR // HOP, voc)
        self.register_buffer("pos", get_sinusoid_encoding_table(frames + 1, d)[None])

    def fft(self, layers, x):
        mask = torch.zeros_like(x[:, :, 0], dtype=torch.bool)
        attn_mask = mask.unsqueeze(1).expand(-1, x.shape[1], -1)
        x = x + self.pos[:, : x.shape[1]]
        for layer in layers:
            x, _ = layer(x, mask=mask, slf_attn_mask=attn_mask)
        return x

    def forward(self, units):
        if not self.voc.one_frame_per_unit:
            units = units_to_frames(units, self.voc)
        va = self.fs2.variance_adaptor
        x = self.fft(self.fs2.encoder.layer_stack, units)
        # phoneme-level pitch and energy, as in VarianceAdaptor.forward with control 1.0
        x = x + va.pitch_embedding(bucketize(va.pitch_predictor(x, None), va.pitch_bins))
        x = x + va.energy_embedding(bucketize(va.energy_predictor(x, None), va.energy_bins))
        log_d = va.duration_predictor(x, None)
        durations = torch.clamp(torch.round(torch.exp(log_d) - 1), min=0).long()[0]
        x = self.fft(self.fs2.decoder.layer_stack, regulate_length(x, durations))
        mel = self.fs2.mel_linear(x)
        mel = self.fs2.postnet(mel) + mel
        return self.vocoder(mel.transpose(1, 2)).squeeze(1)


# ---------------------------------------------------------------- reference (the original code path)

def reference_units(hubert, wav):
    return wn.wav2units(torch.tensor(wav, dtype=torch.float32)[None], hubert, device="cpu")


def reference_wav(fs2, vocoder, units, voc=vocoders.HIFIGAN16K):
    """units2wav() without its int16 conversion."""
    if not voc.one_frame_per_unit:
        units = torch.from_numpy(vocoders.units_to_frames(units[0].numpy(), vocoders.n_frames(units.shape[1], voc), voc))[None]
    n = units.shape[1]
    out = fs2(torch.tensor([0]), units, torch.tensor([n]), n)
    mel = out[1][0, : out[9][0].item()].transpose(0, 1)
    return vocoder(mel[None]).squeeze(1)[0]


# ---------------------------------------------------------------- export and checks

def export(module, example, path, input_name, output_name, axis_names):
    torch.onnx.export(module, (example,), path, opset_version=OPSET, input_names=[input_name],
                      output_names=[output_name],
                      dynamic_axes={input_name: {1: axis_names[0]}, output_name: {1: axis_names[1]}})


def session(path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def cosine(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.mean((a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))))


def snr_db(ref, x):
    ref, x = np.asarray(ref, np.float64), np.asarray(x, np.float64)
    return float(10 * np.log10((ref ** 2).sum() / max(((ref - x) ** 2).sum(), 1e-30)))


# The exported files must match PyTorch to float rounding.
MIN_UNITS_COSINE, MIN_SNR_DB = 0.99999, 60.0


def check_encoder(path, wav, ref_units):
    units = session(path).run(None, {"wav": wav[None]})[0]
    assert units.shape == (1, len(wav) // HOP, 256), f"{path}: units shape {units.shape}"
    cos = cosine(units, ref_units)
    assert cos >= MIN_UNITS_COSINE, f"{path}: units cosine {cos:.6f} vs PyTorch"
    return {"unitsCosine": round(cos, 6)}


def check_decoder(path, units, ref):
    wav = session(path).run(None, {"units": units})[0][0]
    assert len(wav) == len(ref), f"{path}: {len(wav)} samples, PyTorch gives {len(ref)}"
    assert np.isfinite(wav).all(), f"{path}: non-finite samples"
    snr = snr_db(ref, wav)
    assert snr >= MIN_SNR_DB, f"{path}: SNR {snr:.1f} dB vs PyTorch"
    return {"snrDb": round(snr, 1)}


def load_configs():
    """(preprocess_config, model_config): the configs convert.py and client_direct.py use by default."""
    configs = []
    for name in ("my_preprocess16k_LJ.yaml", "my_model16000.yaml"):
        with open(os.path.join(REPO, "config", name)) as f:
            configs.append(yaml.load(f, Loader=yaml.FullLoader))
    return tuple(configs)


def file_info(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return {"path": os.path.basename(path), "bytes": os.path.getsize(path), "sha256": h.hexdigest()}


def test_clips():
    """sample_whisper.wav, and a 3x repeat of it trimmed to a length that isn't a multiple of 320,
    so the checks also cover a second, longer input length than the one traced at export."""
    import soundfile as sf
    wav, sr = sf.read(os.path.join(REPO, "sample_whisper.wav"), dtype="float32")
    assert sr == SR and wav.ndim == 1
    return [wav, np.tile(wav, 3)[: 3 * len(wav) - 123]]


def export_encoder(hubert, eid, out, clips, log):
    refs = [reference_units(hubert, w).numpy() for w in clips]
    path = os.path.join(out, f"encoder-{eid}.onnx")
    export(EncoderExport(hubert).eval(), torch.tensor(clips[0])[None], path, "wav", "units", ("samples", "frames"))
    checks = [check_encoder(path, w, r) for w, r in zip(clips, refs)]
    info = file_info(path)
    log(f"  {info['path']}: {info['bytes'] / 1e6:.0f} MB, {checks}")
    return info, checks


def export_decoder(fs2, vocoder, did, out, units_list, log, voc=vocoders.HIFIGAN16K):
    refs = [reference_wav(fs2, vocoder, torch.tensor(u), voc).numpy() for u in units_list]
    path = os.path.join(out, f"decoder-{did}.onnx")
    export(DecoderExport(fs2, vocoder, voc).eval(), torch.tensor(units_list[0]), path, "units", "wav", ("frames", "samples"))
    checks = [check_decoder(path, u, r) for u, r in zip(units_list, refs)]
    info = file_info(path)
    log(f"  {info['path']}: {info['bytes'] / 1e6:.0f} MB, {checks}")
    return info, checks


DECODERS = {  # id -> label, description, vocoder; the first one found is the app's default
    "sv-narrator": ("Swedish narrator", "A Swedish voice: WESPER's decoder fine-tuned on 14 h of one audiobook narrator (decoder/), with HiFi-GAN 16 kHz", "hifigan16k"),
    "sv-narrator-bigvgan": ("Swedish narrator (BigVGAN)", "The same narrator, trained for NVIDIA's BigVGAN v2 vocoder: 22.05 kHz and clearer, but a 600 MB download, and slow without WebGPU", "bigvgan22k"),
    "googletts": ("English", "WESPER's English voice trained on Google TTS output, with HiFi-GAN 16 kHz", "hifigan16k"),
}


def load_run_configs(run, model_config):
    """(preprocess_config, model_config) of a decoder/train.py run folder, its stats.json found wherever the folder is."""
    with open(os.path.join(run, "preprocess.yaml")) as f:
        pre = yaml.load(f, Loader=yaml.FullLoader)
    pre["path"]["preprocessed_path"] = run
    return pre, model_config


def run_vocoder(preprocess_config):
    """The vocoder a decoder/train.py run was trained for (vocoders.py); WESPER's HiFi-GAN if it doesn't say."""
    return vocoders.spec(preprocess_config.get("vocoder", {}).get("name", vocoders.DEFAULT))


ENCODERS = {  # id -> label, description; the first one found is the app's default
    "sv": ("Swedish", "WESPER's encoder fine-tuned on Swedish Normal2Whisper pairs (colab/)"),
    "original": ("Original", "WESPER's original encoder (English: LibriSpeech, pseudo-whispers, wTIMIT)"),
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default=os.path.join(REPO, "web", "public", "models"))
    p.add_argument("--original", default=f"{RELEASE}/model-layer12-450000.pt", help="original encoder checkpoint")
    p.add_argument("--sv", default=os.path.join(REPO, "colab", "data", "runs", "n2w-finetune", "encoder_best.pt"),
                   help="Swedish encoder checkpoint; skipped if missing, or if set to ''")
    p.add_argument("--sv-decoder", default=os.path.join(REPO, "decoder", "runs", "sv-narrator"),
                   help="Swedish decoder run folder (decoder_best.pt, preprocess.yaml, stats.json); skipped if missing, or if set to ''")
    p.add_argument("--sv-bigvgan-decoder", default=os.path.join(REPO, "decoder", "runs", "sv-narrator-bigvgan22k"),
                   help="the same for the Swedish decoder trained for BigVGAN; skipped if missing, or if set to ''")
    p.add_argument("--fastspeech2", default=f"{RELEASE}/googletts_neutral_best.tar")
    p.add_argument("--hifigan", default=f"{RELEASE}/g_00205000")
    p.add_argument("--timeout", type=float, default=1800, help="seconds before a hung export is aborted")
    args = p.parse_args(argv)
    faulthandler.dump_traceback_later(args.timeout, exit=True)

    out = os.path.abspath(args.out)
    sources = {"original": args.original, "sv": args.sv and os.path.abspath(args.sv)}
    os.makedirs(out, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(REPO)  # FastSpeech2 and HiFi-GAN read their configs by paths relative to the repo
    torch.set_grad_enabled(False)
    log = lambda s: print(s, flush=True)
    try:
        clips = test_clips()
        manifest = {"version": 2, "sampleRate": SR, "hop": HOP, "maxSeconds": MAX_SECONDS,
                    "encoders": [], "decoders": []}
        units_for_decoder = None
        for eid in ("sv", "original"):
            src = sources[eid]
            if not src or (not src.startswith("http") and not os.path.exists(src)):
                log(f"### skipping the {ENCODERS[eid][0]} encoder: no checkpoint at {src!r}")
                continue
            log(f"### encoder {eid}: {src}")
            hubert = wn.load_hubert(src, device="cpu")
            info, checks = export_encoder(hubert, eid, out, clips, log)
            # like MyWhisper2Normal: normalize the input only if the encoder was trained that way
            target = hubert.training_config.get("TARGET_DBFS")
            max_gain = hubert.training_config.get("MAX_GAIN_DB", 40.0) if target is not None else None
            manifest["encoders"].append({
                "id": eid, "label": ENCODERS[eid][0], "description": ENCODERS[eid][1],
                "targetDbfs": target, "maxGainDb": max_gain,
                "source": os.path.relpath(src, REPO) if not src.startswith("http") else src,
                "file": info, "checks": checks})
            if units_for_decoder is None:
                units_for_decoder = [reference_units(hubert, w).numpy() for w in clips]
            del hubert
        if not manifest["encoders"]:
            raise SystemExit("no encoder checkpoints found")

        configs = load_configs()
        loaded = {}  # (vocoder name, checkpoint) -> model, loaded once

        def vocoder_for(voc, checkpoint=None):
            """The vocoder, from a run's own fine-tuned checkpoint (decoder/finetune_vocoder.py) if it has one."""
            if (voc.name, checkpoint) not in loaded:
                if voc.name == "hifigan16k":
                    model = wn.load_hifigan(configs[1], checkpoint_path=args.hifigan, device="cpu")
                else:
                    model = vocoders.load(voc, checkpoint=checkpoint)
                    log(f"  {voc.name}: {fixed_bigvgan_filters(model, voc.n_mels)} anti-aliasing filters given fixed kernels")
                loaded[voc.name, checkpoint] = model
            return loaded[voc.name, checkpoint]

        decoders = []  # (id, configs, checkpoint, vocoder spec, vocoder checkpoint or None)
        for did, folder in (("sv-narrator", args.sv_decoder), ("sv-narrator-bigvgan", args.sv_bigvgan_decoder)):
            run = folder and os.path.abspath(folder)
            if not (run and os.path.exists(os.path.join(run, "decoder_best.pt"))):
                log(f"### skipping the {DECODERS[did][0]} decoder: no decoder_best.pt in {folder!r}")
                continue
            cfg = load_run_configs(run, configs[1])
            voc = run_vocoder(cfg[0])
            if voc.name != DECODERS[did][2]:
                raise SystemExit(f"{run} was trained for {voc.name}, but the {DECODERS[did][0]} voice is for {DECODERS[did][2]}")
            decoders.append((did, cfg, os.path.join(run, "decoder_best.pt"), voc, vocoders.run_checkpoint(cfg[0], run)))
        decoders.append(("googletts", configs, args.fastspeech2, vocoders.HIFIGAN16K, None))
        decoders.sort(key=lambda d: list(DECODERS).index(d[0]))
        for did, cfg, checkpoint, voc, voc_checkpoint in decoders:
            source = [checkpoint if checkpoint.startswith("http") else os.path.relpath(checkpoint, REPO),
                      os.path.relpath(voc_checkpoint, REPO) if voc_checkpoint
                      else args.hifigan if voc.name == "hifigan16k" else voc.checkpoint]
            log(f"### decoder {did}: {source[0]} + {voc.name}")
            fs2 = wn.load_fastspeech2(cfg, checkpoint_path=checkpoint, device="cpu")
            info, checks = export_decoder(fs2, vocoder_for(voc, voc_checkpoint), did, out, units_for_decoder, log, voc)
            manifest["decoders"].append({"id": did, "label": DECODERS[did][0], "description": DECODERS[did][1],
                                         "vocoder": voc.name, "sampleRate": voc.sample_rate, "hop": voc.hop,
                                         "source": source, "file": info, "checks": checks})
            del fs2

        with open(os.path.join(out, "models.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        log(f"### wrote {out}/models.json")
    finally:
        os.chdir(cwd)
        faulthandler.cancel_dump_traceback_later()
    return manifest


if __name__ == "__main__":
    main()
