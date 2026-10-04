"""Export WESPER's models to ONNX for the browser demo in web/.

Writes into web/public/models/ (by default):
  encoder-<id>.onnx         audio [1, T] float32, 16 kHz -> soft units [1, T // 320, 256]
  decoder-<id>.onnx         soft units -> audio [1, S] float32 (FastSpeech2 + HiFi-GAN), one per voice
  models.json               what the app loads: labels, files, sizes, hashes and each encoder's input level

The models stay fp32: fp16 versions were tried and sounded clearly worse in the browser.

Encoders: WESPER's original, and the Swedish fine-tuned one if its checkpoint exists. Decoders
(voices): the Swedish narrator trained by decoder/train.py if its run folder exists, and WESPER's
English googletts one. Every exported file is checked against the PyTorch code path that
convert.py and the GUI use; the script fails if they disagree.

    .venv/bin/python web/export_models.py               # needs: pip install -r web/requirements-export.txt

The wrappers below change how a few operations are written, never what they compute:
- nn.TransformerEncoderLayer's attention is spelled out, because the traced nn.MultiheadAttention
  bakes the sequence length into a Reshape (and its fused fast path has no ONNX export at all).
- torch.bucketize becomes a compare-and-count, and FastSpeech2's length regulator (a Python
  loop over .item() values) becomes a cumsum lookup.
- The decoder's position table is precomputed for 120 s; the original computes the same table
  on the fly beyond 1000 frames (20 s).
"""
import argparse
import faulthandler
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "libs", "FastSpeech2"))
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


class DecoderExport(nn.Module):
    """soft units [1, N, 256] -> wav [1, S]: FastSpeech2's inference path as units2wav() runs it
    (no targets: predicted pitch, energy and durations), then HiFi-GAN. Batch 1, no padding."""

    def __init__(self, fs2, vocoder):
        super().__init__()
        assert isinstance(fs2.encoder.src_word_emb, nn.Identity), "expects 256-dim soft units (soft_unit_dim 256)"
        self.fs2, self.vocoder = fs2, vocoder
        d = fs2.encoder.d_model
        self.register_buffer("pos", get_sinusoid_encoding_table(MAX_SECONDS * SR // HOP + 1, d)[None])

    def fft(self, layers, x):
        mask = torch.zeros_like(x[:, :, 0], dtype=torch.bool)
        attn_mask = mask.unsqueeze(1).expand(-1, x.shape[1], -1)
        x = x + self.pos[:, : x.shape[1]]
        for layer in layers:
            x, _ = layer(x, mask=mask, slf_attn_mask=attn_mask)
        return x

    def forward(self, units):
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


def reference_wav(fs2, vocoder, units):
    """units2wav() without its int16 conversion."""
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


def export_decoder(fs2, vocoder, did, out, units_list, log):
    refs = [reference_wav(fs2, vocoder, torch.tensor(u)).numpy() for u in units_list]
    path = os.path.join(out, f"decoder-{did}.onnx")
    export(DecoderExport(fs2, vocoder).eval(), torch.tensor(units_list[0]), path, "units", "wav", ("frames", "samples"))
    checks = [check_decoder(path, u, r) for u, r in zip(units_list, refs)]
    info = file_info(path)
    log(f"  {info['path']}: {info['bytes'] / 1e6:.0f} MB, {checks}")
    return info, checks


DECODERS = {  # id -> label, description; the first one found is the app's default
    "sv-narrator": ("Swedish narrator", "A Swedish voice: WESPER's decoder fine-tuned on 14 h of one audiobook narrator (decoder/), with HiFi-GAN 16 kHz"),
    "googletts": ("English", "WESPER's English voice trained on Google TTS output, with HiFi-GAN 16 kHz"),
}


def load_run_configs(run, model_config):
    """(preprocess_config, model_config) of a decoder/train.py run folder, its stats.json found wherever the folder is."""
    with open(os.path.join(run, "preprocess.yaml")) as f:
        pre = yaml.load(f, Loader=yaml.FullLoader)
    pre["path"]["preprocessed_path"] = run
    vocoder = pre.get("vocoder", {}).get("name", "hifigan16k")
    if vocoder != "hifigan16k":
        raise SystemExit(f"{run} was trained for {vocoder}: the web app plays decoders for WESPER's 16 kHz HiFi-GAN only")
    return pre, model_config


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
        vocoder = wn.load_hifigan(configs[1], checkpoint_path=args.hifigan, device="cpu")
        decoders = []  # (id, configs, checkpoint, source)
        run = args.sv_decoder and os.path.abspath(args.sv_decoder)
        if run and os.path.exists(os.path.join(run, "decoder_best.pt")):
            decoders.append(("sv-narrator", load_run_configs(run, configs[1]), os.path.join(run, "decoder_best.pt"),
                             [os.path.relpath(os.path.join(run, "decoder_best.pt"), REPO), args.hifigan]))
        else:
            log(f"### skipping the {DECODERS['sv-narrator'][0]} decoder: no decoder_best.pt in {args.sv_decoder!r}")
        decoders.append(("googletts", configs, args.fastspeech2, [args.fastspeech2, args.hifigan]))
        for did, cfg, checkpoint, source in decoders:
            log(f"### decoder {did}: {source[0]} + {args.hifigan}")
            fs2 = wn.load_fastspeech2(cfg, checkpoint_path=checkpoint, device="cpu")
            info, checks = export_decoder(fs2, vocoder, did, out, units_for_decoder, log)
            manifest["decoders"].append({"id": did, "label": DECODERS[did][0], "description": DECODERS[did][1],
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
