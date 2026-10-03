"""Prepare a single speaker's recordings (e.g. an audiobook) for training a WESPER decoder.

The decoder (FastSpeech2) turns speech units into a mel spectrogram, which a vocoder turns
into audio. Training it on one speaker gives WESPER that speaker's voice. It needs no
transcripts, only audio. The decoder is trained for one vocoder (--vocoder, see vocoders.py):
hifigan16k, WESPER's own and the default, or bigvgan22k (NVIDIA BigVGAN v2). For every recording
this script:

1. Decodes it to 16 kHz mono and cuts it into utterances of MIN_SECONDS-MAX_SECONDS at pauses.
2. Normalizes each utterance to TARGET_DBFS speech level, as WESPER's input is normalized.
3. Computes:
   - units: WESPER's original encoder's speech units (one per 20 ms), the decoder's input. The
     fine-tuned Swedish encoder was trained to produce these same units from whispers.
   - mel: the training target, computed as the vocoder computes its own training mels, from the
     recording decoded at the vocoder's sample rate. For hifigan16k its frames line up one to
     one with the units; other vocoders have other frame rates, and the units are interpolated
     to them when training (vocoders.units_to_frames).
   - pitch (Hz) and energy at each mel frame, which the decoder learns to predict.

Output in OUT_DIR:
  segments/<id>.npz  units (float16, one per 20 ms), mel (float16), pitch, energy (float32):
                     the last three T mel frames (for hifigan16k, units are T long too)
  audio/<id>.flac    the utterance as cut from the recording, before normalization
  segments.tsv       id, recording, split, start/end seconds, frames, speech level
  stats.json         pitch and energy normalization in FastSpeech2's format
  prep.json          the settings used

The last --val-recordings recordings become the validation split. Recordings that are already
done are skipped when the script is re-run. The source folder is only read.

usage: python decoder/prepare_data.py AUDIO_DIR OUT_DIR [--vocoder hifigan16k] [--val-recordings 3] [--device cpu]
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from colab.prepare_data import speech_dbfs  # noqa: E402  (the encoder pipeline's level measure)
import vocoders  # noqa: E402

SR = 16000  # the encoder's sample rate
HOP = 320  # 20 ms: one unit
MAX_FRAMES = 1000  # FastSpeech2's max_seq_len (config/my_model16000.yaml): mel frames per utterance
TARGET_DBFS = -20.0
MAX_GAIN_DB = 40.0
MIN_SPEECH_DBFS = -45.0  # quieter recordings are only noise, as in the encoder's data
MIN_SECONDS, MAX_SECONDS = 1.0, 12.0
PAUSE_SECONDS = 0.25  # cut only in pauses at least this long...
SILENCE_DB = 30.0  # ...where the level is this far below the speech level
EDGE_SECONDS = 0.1  # silence kept before and after each utterance
ORIGINAL_ENCODER = "https://github.com/rkmt/wesper-demo/releases/download/v0.1/model-layer12-450000.pt"
AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a", ".ogg")


def decode(path, sample_rate=SR):
    """Any audio file as mono float32."""
    pcm = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sample_rate),
                          "-f", "f32le", "pipe:1"], check=True, capture_output=True, timeout=600).stdout
    return np.frombuffer(pcm, dtype=np.float32).copy()


def frame_db(x):
    """Level of each 20 ms frame in dBFS."""
    n = len(x) // HOP
    return 10 * np.log10(np.mean(x[: n * HOP].astype(np.float64).reshape(n, HOP) ** 2, axis=1) + 1e-12)


def segment(x, min_s=MIN_SECONDS, max_s=MAX_SECONDS):
    """(start, end) samples of utterances: cut in pauses, MIN to MAX seconds, edges trimmed.

    Pauses are runs of at least PAUSE_SECONDS more than SILENCE_DB below the speech level. Cuts
    go in the middle of a pause, as late as fits within MAX seconds. A stretch with no pause for
    MAX seconds is cut at its quietest frame.
    """
    db = frame_db(x)
    n = len(db)
    level = speech_dbfs(x)
    quiet = db < level - SILENCE_DB
    if level < MIN_SPEECH_DBFS or quiet.all():
        return []
    pause_frames = int(PAUSE_SECONDS * SR / HOP)
    cuts, i = [], 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if j - i >= pause_frames:
                cuts.append((i + j) // 2)
            i = j
        else:
            i += 1
    first, last = int(np.argmax(~quiet)), n - int(np.argmax(~quiet[::-1]))
    candidates = [c for c in cuts if first < c < last] + [last]
    min_f, max_f = int(min_s * SR / HOP), int(max_s * SR / HOP)

    bounds, start = [], first
    while start < last:
        fitting = [c for c in candidates if start + min_f <= c <= start + max_f]
        if last - start <= max_f:
            end = last
        elif fitting:
            end = fitting[-1]
        else:  # no usable pause: cut at the quietest frame in range
            end = start + min_f + int(np.argmin(db[start + min_f:start + max_f]))
        bounds.append((start, end))
        start = end
        while start < last and quiet[start]:  # skip the rest of the pause
            start += 1

    # Trim each utterance to its speech plus up to EDGE_SECONDS of silence on each side. The
    # padding never crosses a cut point, so utterances can't overlap; the first and last may
    # take silence from the start and end of the recording.
    bounds[0] = (0, bounds[0][1])
    bounds[-1] = (bounds[-1][0], n)
    edge = int(EDGE_SECONDS * SR / HOP)
    out = []
    for s, e in bounds:
        loud = np.flatnonzero(~quiet[s:e])
        if len(loud) == 0:
            continue
        s2, e2 = max(s + loud[0] - edge, s), min(s + loud[-1] + 1 + edge, e)
        if (e2 - s2) * HOP >= min_s * SR:
            out.append((s2 * HOP, e2 * HOP))
    return out


def normalize(x):
    gain_db = min(TARGET_DBFS - speech_dbfs(x), MAX_GAIN_DB)
    return (x * 10 ** (gain_db / 20)).astype(np.float32), gain_db


def mel_energy(x, voc=vocoders.HIFIGAN16K):
    """Log-mel spectrogram and energy as the vocoder's training mels (see vocoders.mel_energy)."""
    return vocoders.mel_energy(x, voc)


def pitch(x, n_frames, frame_seconds=HOP / SR):
    """F0 in Hz at each mel frame's center, with unvoiced stretches filled by linear interpolation.

    WORLD's DIO + StoneMask on 16 kHz audio, as in FastSpeech2's preprocessing, at 10 ms steps,
    then sampled at the frame centers (t + 0.5) * frame_seconds. All-unvoiced input gives zeros.
    """
    import pyworld
    x = np.asarray(x, dtype=np.float64)
    f0, t = pyworld.dio(x, SR, frame_period=10.0)
    f0 = pyworld.stonemask(x, f0, t, SR)
    voiced = f0 > 0
    if voiced.sum() < 2:
        return np.zeros(n_frames, dtype=np.float32)
    f0 = np.interp(t, t[voiced], f0[voiced])
    centers = (np.arange(n_frames) + 0.5) * frame_seconds
    return np.interp(centers, t, f0).astype(np.float32)


def remove_outliers(values):
    """Values strictly within 1.5 interquartile ranges of the quartiles (FastSpeech2's remove_outlier)."""
    p25, p75 = np.percentile(values, [25, 75])
    low, high = p25 - 1.5 * (p75 - p25), p75 + 1.5 * (p75 - p25)
    return values[(values > low) & (values < high)]


def compute_stats(pitches, energies):
    """FastSpeech2's stats.json, computed as its preprocessor does (and WESPER's did).

    Mean and std are fitted on every utterance's values with that utterance's outliers removed.
    The entry is [min, max] of all values once normalized, then [mean, std].
    """
    stats = {}
    for name, per_utterance in (("pitch", pitches), ("energy", energies)):
        per_utterance = [np.asarray(v, dtype=np.float64) for v in per_utterance]
        fitted = np.concatenate([remove_outliers(v) for v in per_utterance])
        mean, std = float(fitted.mean()), float(fitted.std())
        normalized = (np.concatenate(per_utterance) - mean) / std
        stats[name] = [float(normalized.min()), float(normalized.max()), mean, std]
    return stats


def load_encoder(path_or_url, device):
    import torch
    from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
    from libs.hubert.model import HubertSoft
    if path_or_url.startswith("http"):
        cached = os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(path_or_url))
        if not os.path.exists(cached):
            os.makedirs(os.path.dirname(cached), exist_ok=True)
            torch.hub.download_url_to_file(path_or_url, cached)
        path_or_url = cached
    state = torch.load(path_or_url, map_location="cpu")["hubert"]
    consume_prefix_in_state_dict_if_present(state, "module.")
    encoder = HubertSoft()
    encoder.load_state_dict(state, strict=True)
    return encoder.eval().to(device)


def units(encoder, x, device):
    import torch
    with torch.inference_mode():
        return encoder.units(torch.from_numpy(x)[None, None].to(device))[0].float().cpu().numpy()


def max_seconds(voc):
    """The longest utterance: MAX_SECONDS, or less if the vocoder's frames wouldn't fit in MAX_FRAMES."""
    return min(MAX_SECONDS, 0.98 * MAX_FRAMES * voc.frame_seconds)


def prepare_recording(path, out_dir, encoder, device, voc=vocoders.HIFIGAN16K):
    """Segment one recording and write its utterances. Returns their manifest rows."""
    name = os.path.splitext(os.path.basename(path))[0]
    x = decode(path)  # 16 kHz: cutting, units and pitch
    xv = x if voc.sample_rate == SR else decode(path, voc.sample_rate)  # the vocoder's rate: mel and energy
    rows = []
    for k, (s, e) in enumerate(segment(x, max_s=max_seconds(voc))):
        seg_id = f"{name}_{k:04d}"
        raw = x[s:e]
        wav, gain_db = normalize(raw)
        u = units(encoder, wav, device)
        if voc.sample_rate == SR:
            wav_v = wav
        else:  # the same stretch of the recording, at the vocoder's rate, with the same gain
            sv, ev = (round(i * voc.sample_rate / SR) for i in (s, e))
            wav_v = (xv[sv:ev] * 10 ** (gain_db / 20)).astype(np.float32)
        mel, energy = mel_energy(wav_v, voc)
        n = min(mel.shape[1], vocoders.n_frames(len(u), voc))
        if voc.one_frame_per_unit:
            u = u[:n]
        f0 = pitch(wav, n, voc.frame_seconds)
        np.savez(os.path.join(out_dir, "segments", seg_id + ".npz"), units=u.astype(np.float16),
                 mel=mel[:, :n].astype(np.float16), pitch=f0, energy=energy[:n].astype(np.float32))
        sf.write(os.path.join(out_dir, "audio", seg_id + ".flac"), np.clip(raw, -1, 1), SR, subtype="PCM_16")
        rows.append({"id": seg_id, "recording": name, "start": f"{s / SR:.2f}", "end": f"{e / SR:.2f}",
                     "frames": n, "speech_dbfs": f"{speech_dbfs(raw):.1f}", "gain_db": f"{gain_db:.1f}"})
    return rows


FIELDS = ["id", "recording", "split", "start", "end", "frames", "speech_dbfs", "gain_db"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("audio_dir", help="folder of one speaker's recordings (only read)")
    parser.add_argument("out_dir", help="output folder")
    parser.add_argument("--val-recordings", type=int, default=3, help="last N recordings (sorted by name) for validation")
    parser.add_argument("--limit", type=int, help="only use the first N recordings (for quick tests)")
    parser.add_argument("--vocoder", default=vocoders.DEFAULT, help="hifigan16k, bigvgan22k or bigvgan:<model> (vocoders.py)")
    parser.add_argument("--encoder", default=ORIGINAL_ENCODER, help="encoder that computes the units")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    args = parser.parse_args()

    src, out = os.path.realpath(args.audio_dir), os.path.realpath(args.out_dir)
    if (out + os.sep).startswith(src + os.sep):
        sys.exit("OUT_DIR must not be inside AUDIO_DIR")
    recordings = sorted(p for p in glob.glob(os.path.join(src, "*")) if p.lower().endswith(AUDIO_EXTENSIONS))
    if not recordings:
        sys.exit(f"no audio files in {src}")
    val = {os.path.splitext(os.path.basename(p))[0] for p in recordings[-args.val_recordings:]} if args.val_recordings else set()
    recordings = recordings[: args.limit]

    voc = vocoders.spec(args.vocoder)
    previous = os.path.join(out, "prep.json")
    if os.path.exists(previous):  # a re-run reuses finished recordings, so they must be for the same vocoder
        with open(previous) as f:
            prepared_for = json.load(f).get("vocoder", vocoders.DEFAULT)
        if prepared_for != voc.name:
            sys.exit(f"{out} was prepared for {prepared_for}, not {voc.name}: use another OUT_DIR")
    for sub in ("segments", "audio", "done"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    encoder, started = None, time.time()
    for i, path in enumerate(recordings, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        done = os.path.join(out, "done", name + ".tsv")
        if os.path.exists(done):
            continue
        if encoder is None:  # only when there's work left: a re-run over finished data is quick
            encoder = load_encoder(args.encoder, args.device)
        rows = prepare_recording(path, out, encoder, args.device, voc)
        with open(done + ".tmp", "w", newline="") as f:  # rename after writing: a crash can't leave half a list
            writer = csv.DictWriter(f, FIELDS, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(done + ".tmp", done)
        print(f"{i}/{len(recordings)} {name}: {len(rows)} utterances, {time.time() - started:.0f} s", flush=True)

    # The manifest and stats are rebuilt from all finished recordings on every run.
    rows, pitches, energies, unvoiced = [], [], [], 0
    for path in recordings:
        name = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(out, "done", name + ".tsv"), newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                with np.load(os.path.join(out, "segments", r["id"] + ".npz")) as d:
                    if not (d["pitch"] > 0).any():  # under 2 voiced frames: dropped, as FastSpeech2 does
                        unvoiced += 1
                        continue
                    pitches.append(d["pitch"])
                    energies.append(d["energy"])
                rows.append({**r, "split": "val" if name in val else "train"})
    with open(os.path.join(out, "segments.tsv"), "w", newline="") as f:
        writer = csv.DictWriter(f, FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    if unvoiced:
        print(f"left out {unvoiced} utterances with no voiced frames")

    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(compute_stats(pitches, energies), f)
    with open(os.path.join(out, "prep.json"), "w") as f:
        json.dump({"vocoder": voc.name, "sample_rate": voc.sample_rate, "hop": voc.hop, "n_mels": voc.n_mels,
                   "unit_sample_rate": SR, "unit_hop": HOP, "target_dbfs": TARGET_DBFS, "encoder": args.encoder,
                   "min_seconds": MIN_SECONDS, "max_seconds": round(max_seconds(voc), 2), "mel": "hifigan"}, f, indent=1)

    for split in ("train", "val"):
        part = [r for r in rows if r["split"] == split]
        hours = sum(int(r["frames"]) for r in part) * voc.frame_seconds / 3600
        print(f"{split}: {len(part)} utterances, {hours:.2f} h, from {len({r['recording'] for r in part})} recordings")


if __name__ == "__main__":
    main()
