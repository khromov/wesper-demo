"""Preview BigVGAN with WESPER's existing decoders, before training a decoder for it.

WESPER's decoders output mels for its 16 kHz HiFi-GAN (vocoders.HIFIGAN16K). BigVGAN-22k
(bigvgan22k) uses the same 80 mel bands over 0-8 kHz, but at 22.05 kHz with 11.6 ms frames, so
a HiFi-GAN mel can be turned into an approximate BigVGAN mel:

1. resample its frames in time to BigVGAN's frame centers (vocoders.units_to_frames: HiFi-GAN
   frames sit exactly where the units do), and
2. correct each band's level with a straight line, mel22 ~ a * mel16 + c, fitted on real
   recordings analyzed both ways ("fit").

The two analyses use different window lengths, so this is only an approximation; a decoder
trained for BigVGAN (decoder/prepare_data.py --vocoder bigvgan22k) does better. Commands:

  fit       fits the per-band correction on Common Voice recordings; writes OUT/mel_map.json and
            reports how close the corrected mels come to real BigVGAN mels on held-out clips.
  whispers  the encoder evaluation's held-out whispers (colab/evaluate.py's OUT_DIR) through the
            fine-tuned encoder, WESPER's Google TTS decoder and BigVGAN: OUT/wesper-ft-bigvgan/.
  narrator  copy synthesis of prepared narrator utterances (decoder/prepare_data.py data): her
            real mel through HiFi-GAN, her real BigVGAN mel through BigVGAN, and her HiFi-GAN mel
            corrected through BigVGAN: OUT/narrator/<kind>/.

usage:
  python decoder/bigvgan_preview.py fit CORPUS_DIR OUT [--clips 300]
  python decoder/bigvgan_preview.py whispers EVAL_DIR ENCODER OUT [--limit N]
  python decoder/bigvgan_preview.py narrator AUDIO_DIR DATA_DIR OUT [--count 8]
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import vocoders  # noqa: E402
from colab.prepare_data import speech_dbfs  # noqa: E402

HIFI = vocoders.HIFIGAN16K
RELEASE = "https://github.com/rkmt/wesper-demo/releases/download/v0.1"


def decode(path, sample_rate, start=None, end=None):
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}", "-to", f"{end:.3f}"]
    pcm = subprocess.run(cmd + ["-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1"],
                         check=True, capture_output=True, timeout=300).stdout
    return np.frombuffer(pcm, dtype=np.float32).copy()


def to_frames(mel16, big, n_out):
    """A HiFi-GAN mel (80 x T, 20 ms frames) at BigVGAN's frame centers: 80 x n_out."""
    return vocoders.units_to_frames(np.ascontiguousarray(mel16.T), n_out, big).T


def fit_band_map(pairs):
    """Per-band least squares mel22 ~ a * mel16 + c over all frames of (mel16 at BigVGAN frames, mel22) pairs."""
    x = np.concatenate([p[0] for p in pairs], axis=1).astype(np.float64)
    y = np.concatenate([p[1] for p in pairs], axis=1).astype(np.float64)
    a, c = np.empty(len(x)), np.empty(len(x))
    for b in range(len(x)):
        a[b], c[b] = np.polyfit(x[b], y[b], 1)
    return a, c


def apply_band_map(mel, a, c):
    return (np.asarray(a)[:, None] * mel + np.asarray(c)[:, None]).astype(np.float32)


def both_mels(x16, x22, big):
    """(HiFi-GAN mel at BigVGAN's frames, BigVGAN mel) of one recording decoded at 16 and 22.05 kHz."""
    norm = 10 ** ((-20 - speech_dbfs(x16)) / 20)  # the level WESPER's audio is at
    mel22, _ = vocoders.mel_energy(x22 * norm, big)
    mel16, _ = vocoders.mel_energy(x16 * norm, HIFI)
    n = min(mel22.shape[1], vocoders.n_frames(mel16.shape[1], big))
    return to_frames(mel16, big, n), mel22[:, :n]


def load_map(out):
    with open(os.path.join(out, "mel_map.json")) as f:
        m = json.load(f)
    return np.array(m["a"]), np.array(m["c"])


def fit(args):
    big = vocoders.spec("bigvgan22k")
    with open(os.path.join(args.corpus, "train.tsv"), newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    rows = rows[:: max(1, len(rows) // (args.clips + 50))][: args.clips + 50]  # spread over speakers
    pairs = []
    for r in rows:
        path = os.path.join(args.corpus, "clips", r["path"])
        x16, x22 = decode(path, 16000), decode(path, big.sample_rate)
        if speech_dbfs(x16) > -45:
            pairs.append(both_mels(x16, x22, big))
    held, used = pairs[: 50], pairs[50:]
    a, c = fit_band_map(used)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "mel_map.json"), "w") as f:
        json.dump({"from": "hifigan16k", "to": big.name, "clips": len(used), "a": a.tolist(), "c": c.tolist()}, f)
    before = np.mean([np.abs(m16 - m22).mean() for m16, m22 in held])
    after = np.mean([np.abs(apply_band_map(m16, a, c) - m22).mean() for m16, m22 in held])
    print(f"fitted on {len(used)} clips. Held-out mean |error| vs a real BigVGAN mel: "
          f"{before:.3f} frames only -> {after:.3f} with the band correction")


def whispers(args):
    import torch
    os.chdir(REPO)
    import whisper_normal as wn
    big = vocoders.spec("bigvgan22k")
    a, c = load_map(args.out)
    w2n = wn.MyWhisper2Normal(argparse.Namespace(
        preprocess_config="config/my_preprocess16k_LJ.yaml", model_config="config/my_model16000.yaml", device="cpu",
        hubert=args.encoder, fastspeech2=f"{RELEASE}/googletts_neutral_best.tar", hifigan=f"{RELEASE}/g_00205000"),
        load_vocoder=False)
    vocoder = vocoders.load(big)
    with open(os.path.join(args.eval_dir, "reference.tsv"), newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))[: args.limit]
    folder = os.path.join(args.out, "wesper-ft-bigvgan")
    os.makedirs(folder, exist_ok=True)
    for i, r in enumerate(rows, 1):
        dst = os.path.join(folder, r["clip"] + ".wav")
        if os.path.exists(dst):
            continue
        x, _ = sf.read(os.path.join(args.eval_dir, "whisper-rec", r["clip"] + ".wav"), dtype="float32")
        units = w2n.wav2units(torch.from_numpy(wn.normalize_level(x, w2n.target_dbfs, w2n.max_gain_db)
                                               if w2n.target_dbfs is not None else x)[None])
        with torch.no_grad():
            out = w2n.fs2model(torch.tensor([0]), units, torch.tensor([units.shape[1]]), units.shape[1])
        mel16 = out[1][0].T.numpy()  # what WESPER's HiFi-GAN would get
        mel = apply_band_map(to_frames(mel16, big, vocoders.n_frames(mel16.shape[1], big)), a, c)
        sf.write(dst + ".tmp.wav", vocoders.synthesize(vocoder, mel), big.sample_rate, subtype="PCM_16")
        os.replace(dst + ".tmp.wav", dst)
        if i % 25 == 0 or i == len(rows):
            print(f"{i}/{len(rows)} clips", flush=True)


def narrator(args):
    big = vocoders.spec("bigvgan22k")
    a, c = load_map(args.out)
    hifi, bigv = vocoders.load(HIFI), vocoders.load(big)
    with open(os.path.join(args.data, "segments.tsv"), newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["split"] == "val"][: args.count]
    kinds = ("recording", "hifigan", "bigvgan", "bigvgan-from-hifigan-mel")
    for k in kinds:
        os.makedirs(os.path.join(args.out, "narrator", k), exist_ok=True)
    for r in rows:
        src = os.path.join(args.audio, r["recording"] + ".mp3")
        start, end = float(r["start"]), float(r["end"])
        x16, x22 = decode(src, 16000, start, end), decode(src, big.sample_rate, start, end)
        gain = 10 ** (float(r["gain_db"]) / 20)
        mel16, _ = vocoders.mel_energy(x16 * gain, HIFI)
        mel22, _ = vocoders.mel_energy(x22 * gain, big)
        n = min(mel22.shape[1], vocoders.n_frames(mel16.shape[1], big))
        out = {"recording": (x22 * gain, big.sample_rate),
               "hifigan": (vocoders.synthesize(hifi, mel16), HIFI.sample_rate),
               "bigvgan": (vocoders.synthesize(bigv, mel22), big.sample_rate),
               "bigvgan-from-hifigan-mel": (vocoders.synthesize(bigv, apply_band_map(to_frames(mel16, big, n), a, c)),
                                            big.sample_rate)}
        for k in kinds:
            sf.write(os.path.join(args.out, "narrator", k, r["id"] + ".wav"), out[k][0], out[k][1], subtype="FLOAT")
    print(f"wrote {len(rows)} utterances x {len(kinds)} versions to {os.path.join(args.out, 'narrator')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fit")
    p.add_argument("corpus", help="Common Voice folder (train.tsv, clips/); only read")
    p.add_argument("out")
    p.add_argument("--clips", type=int, default=300)
    p = sub.add_parser("whispers")
    p.add_argument("eval_dir", help="colab/evaluate.py convert output (reference.tsv, whisper-rec/)")
    p.add_argument("encoder", help="fine-tuned encoder, e.g. encoder_best.pt")
    p.add_argument("out")
    p.add_argument("--limit", type=int)
    p = sub.add_parser("narrator")
    p.add_argument("audio", help="the recordings folder, e.g. swe-audiobook/book1")
    p.add_argument("data", help="decoder/prepare_data.py output for them")
    p.add_argument("out")
    p.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    for k in ("corpus", "out", "eval_dir", "encoder", "audio", "data"):
        if getattr(args, k, None):
            setattr(args, k, os.path.realpath(getattr(args, k)))
    {"fit": fit, "whispers": whispers, "narrator": narrator}[args.command](args)


if __name__ == "__main__":
    main()
