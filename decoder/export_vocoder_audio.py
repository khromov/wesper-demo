"""Export each prepared utterance as audio at its vocoder's sample rate, for fine-tuning the vocoder.

decoder/prepare_data.py computes an utterance's mel targets from the recording decoded at the
vocoder's sample rate, but keeps the utterance's audio only at 16 kHz. Fine-tuning the vocoder
on the decoder's output (decoder/finetune_vocoder.py) needs the matching audio at the vocoder's
rate. This rebuilds it exactly as prepare_data.py cut it:

  - the same stretch of the recording: segments.tsv's start and end, which are exact multiples
    of 20 ms;
  - the same gain: recomputed from the 16 kHz cut, as prepare_data.py computed it (segments.tsv
    rounds it to 0.1 dB);
  - trimmed to the utterance's mel frames: frame t is samples t * hop to (t + 1) * hop.

Computing the mel of the result (vocoders.mel_energy) gives the stored target again.

Output in OUT_DIR:
  <id>.flac     the utterance at the vocoder's rate, 24-bit, HEADROOM_DB below its level in
                training, so peaks don't clip; load() adds the headroom back. (16-bit would
                change the mel of near-silent stretches by up to 1.3.)
  export.json   the vocoder, sample rate, hop and headroom

Utterances that are already exported are skipped when the script is re-run. AUDIO_DIR and
DATA_DIR are only read.

usage: python decoder/export_vocoder_audio.py AUDIO_DIR DATA_DIR OUT_DIR
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from decoder import prepare_data  # noqa: E402

HEADROOM_DB = 12.0  # as the encoder's data (colab/prepare_data.py); normalized speech peaks well below +12 dBFS


def find_recording(audio_dir, name):
    for ext in prepare_data.AUDIO_EXTENSIONS:
        path = os.path.join(audio_dir, name + ext)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"no recording {name!r} in {audio_dir}")


def cut(x16, xv, row, sample_rate, hop):
    """One utterance at the vocoder's rate, at its training level, frames * hop samples long."""
    sr = prepare_data.SR
    s, e = round(float(row["start"]) * sr), round(float(row["end"]) * sr)
    _, gain_db = prepare_data.normalize(x16[s:e])
    sv, ev = (round(i * sample_rate / sr) for i in (s, e))
    n = int(row["frames"]) * hop
    wav = (xv[sv:ev] * 10 ** (gain_db / 20)).astype(np.float32)[:n]
    if len(wav) < n:
        raise ValueError(f"{row['id']}: {len(wav)} samples, but its {row['frames']} frames need {n}")
    return wav


def load(path, headroom_db=HEADROOM_DB):
    """An exported utterance as float32 at its training level."""
    wav, sample_rate = sf.read(path, dtype="float32")
    return wav * 10 ** (headroom_db / 20), sample_rate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("audio_dir", help="the recordings prepare_data.py was run on (only read)")
    parser.add_argument("data_dir", help="prepare_data.py's output for the vocoder (only read)")
    parser.add_argument("out_dir", help="output folder")
    args = parser.parse_args(argv)

    with open(os.path.join(args.data_dir, "prep.json")) as f:
        prep = json.load(f)
    sample_rate, hop = prep["sample_rate"], prep["hop"]
    with open(os.path.join(args.data_dir, "segments.tsv"), newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    os.makedirs(args.out_dir, exist_ok=True)
    by_recording = defaultdict(list)
    for r in rows:
        if not os.path.exists(os.path.join(args.out_dir, r["id"] + ".flac")):
            by_recording[r["recording"]].append(r)

    started, scale, peak, clipped = time.time(), 10 ** (-HEADROOM_DB / 20), 0.0, 0
    for i, (name, todo) in enumerate(sorted(by_recording.items()), 1):
        path = find_recording(args.audio_dir, name)
        x16 = prepare_data.decode(path)
        xv = x16 if sample_rate == prepare_data.SR else prepare_data.decode(path, sample_rate)
        for r in todo:
            wav = cut(x16, xv, r, sample_rate, hop) * scale
            peak = max(peak, float(np.abs(wav).max()))
            clipped += int((np.abs(wav) > 1).sum())
            out = os.path.join(args.out_dir, r["id"] + ".flac")
            sf.write(out + ".tmp", np.clip(wav, -1, 1), sample_rate, subtype="PCM_24", format="FLAC")
            os.replace(out + ".tmp", out)  # rename after writing: a crash can't leave half a file
        print(f"{i}/{len(by_recording)} {name}: {len(todo)} utterances, {time.time() - started:.0f} s", flush=True)

    with open(os.path.join(args.out_dir, "export.json"), "w") as f:
        json.dump({"vocoder": prep.get("vocoder", "hifigan16k"), "sample_rate": sample_rate, "hop": hop,
                   "headroom_db": HEADROOM_DB, "utterances": len(rows)}, f, indent=1)
    done = sum(len(t) for t in by_recording.values())
    print(f"exported {done} utterances ({len(rows) - done} already there)"
          + (f", stored peak {20 * np.log10(peak):.1f} dBFS" if done else "") + (f", {clipped} samples clipped" if clipped else ""))


if __name__ == "__main__":
    main()
