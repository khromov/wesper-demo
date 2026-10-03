"""Prepare Swedish Common Voice clips and their Normal2Whisper versions for training on Colab.

Each selected clip becomes a pair of 16 kHz mono FLAC files, normal/<clip>.flac and
whisper/<clip>.flac, processed exactly as the encoder will see them:

1. A touchpad thud at the start (the click that started the recording) is muted in both
   versions. See thud_end().
2. Each version is normalized to TARGET_DBFS speech level. Common Voice levels vary by 30+ dB
   between clips, and WESPER's encoder is very level-sensitive.
3. Files are stored HEADROOM_DB below that level, so that 16-bit FLAC doesn't clip peaky
   clips. The notebook adds HEADROOM_DB back when loading.

Clips whose speech is quieter than MIN_SPEECH_DBFS are mostly noise and are skipped.

Everything goes into OUT_TAR with manifest.tsv (split, speaker, levels, thud length, sentence)
and prep.json (the settings above). With --export-dir, the same files are also written there
as a folder tree, for listening in the corpus player.

Splits are speaker-disjoint. "train" is every validated clip whose speaker is not in
Common Voice's test set. "val" is a sample of test-set clips, one per speaker in turn, so
validation measures voices the encoder never trained on. The remaining test clips are left out.

The corpus and the Normal2Whisper output are only read.

usage: python colab/prepare_data.py CORPUS_DIR N2W_CLIPS_DIR OUT_TAR [--export-dir DIR] [--limit N]
"""
import argparse
import csv
import io
import json
import os
import random
import subprocess
import sys
import tarfile
import time

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

SR = 16000
HOP = 320  # 20 ms frames, the encoder's frame size

TARGET_DBFS = -20.0  # Common Voice's median speech level
HEADROOM_DB = 12.0  # stored this far below TARGET_DBFS; normalized whispers peak up to ~+13 dBFS
MAX_GAIN_DB = 40.0  # never amplify more than this
MIN_SPEECH_DBFS = -45.0  # quieter clips are mostly noise and are skipped
THUD_MAX_MS = 200  # a leading sound longer than this is treated as speech and kept
THUD_GAP_MS = 40  # ...and it only counts as a thud if this much quiet follows it


def speech_dbfs(x):
    """Speech level in dBFS of float audio. Keep in sync with speech_dbfs() in the notebook.

    Frames more than 30 dB below the loudest ones count as silence and are dropped. The level
    is the mean power of the remaining 20 ms frames within 20 dB of their 90th percentile, with
    each frame capped at that percentile. Leading/trailing silence, background noise and
    clicks barely move it.
    """
    n = len(x) // HOP
    if n == 0:
        return -120.0
    energy = np.mean(x[: n * HOP].astype(np.float64).reshape(n, HOP) ** 2, axis=1)
    speech = energy[energy >= np.percentile(energy, 99) / 1000]
    ref = np.percentile(speech, 90)
    active = np.minimum(speech[speech >= ref / 100], ref)
    return float(10 * np.log10(active.mean() + 1e-12))


def thud_end(x, level_db):
    """Sample index where a leading touchpad thud ends, or 0 if the clip doesn't start with one.

    Many clips start with the click that started the recording: a short transient right after
    the MP3's leading digital silence, then quiet before the speech. It counts as a thud if the
    first sound ends within THUD_MAX_MS and THUD_GAP_MS of quiet follows. Speech that starts
    straight away has no such gap, so it is left alone.
    """
    nonzero = np.flatnonzero(x)
    if len(nonzero) == 0:
        return 0
    first, frame = nonzero[0], SR // 200  # 5 ms frames
    n = (len(x) - first) // frame
    if n == 0:
        return 0
    energy = 10 * np.log10(np.mean(x[first:first + n * frame].astype(np.float64).reshape(n, frame) ** 2, axis=1) + 1e-12)
    # Quiet means near the background noise, and at least 45 dB below the speech.
    quiet = energy <= max(np.percentile(energy, 10) + 10, level_db - 45)
    gap = THUD_GAP_MS // 5
    for i in range(1, min(THUD_MAX_MS // 5, n - gap) + 1):
        if quiet[i:i + gap].all():
            return int(first + i * frame) if not quiet[:i].all() else 0
    return 0


def process(normal, whisper):
    """Mute a leading thud in both versions, then bring each to the stored speech level.

    Takes and returns float audio. The pair stays sample-aligned, since Normal2Whisper
    preserves timing. Also returns details for the manifest.
    """
    n = min(len(normal), len(whisper))  # Normal2Whisper adds a few ms at the end
    normal, whisper = normal[:n].copy(), whisper[:n].copy()
    normal_db, whisper_db = speech_dbfs(normal), speech_dbfs(whisper)
    cut = thud_end(normal, normal_db)
    if cut:
        fade = np.linspace(0, 1, SR // 200, dtype=np.float32)  # 5 ms fade-in after the muted part
        for x in (normal, whisper):
            x[:cut] = 0
            x[cut:cut + len(fade)] *= fade[: len(x) - cut]
    out, clipped = [], 0
    for x in (normal, whisper):
        gain_db = min(TARGET_DBFS - speech_dbfs(x), MAX_GAIN_DB) - HEADROOM_DB
        y = x * 10 ** (gain_db / 20)
        clipped += int((np.abs(y) > 1).sum())
        out.append(np.clip(y, -1, 1).astype(np.float32))
    info = {"normal_dbfs": normal_db, "whisper_dbfs": whisper_db, "thud_ms": cut * 1000 / SR, "clipped": clipped}
    return out[0], out[1], info


def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def flac_bytes(samples):
    buf = io.BytesIO()
    sf.write(buf, samples, SR, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def prepare_pair(job):
    """Return processed FLAC bytes for one clip's normal and whispered versions, plus details."""
    mp3, wav = job
    # Same resampling as the toWhisper run: ffmpeg to 16 kHz mono 16-bit.
    pcm = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", mp3, "-ac", "1", "-ar", str(SR),
         "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"],
        check=True, capture_output=True, timeout=120,  # a clip takes milliseconds; never hang the build
    ).stdout
    normal = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
    whisper, sr = sf.read(wav, dtype="float32")
    if sr != SR or whisper.ndim != 1:
        raise ValueError(f"{wav}: expected {SR} Hz mono, got {sr} Hz, shape {whisper.shape}")
    if speech_dbfs(normal) < MIN_SPEECH_DBFS:
        return None
    normal, whisper, info = process(normal, whisper)
    return flac_bytes(normal), flac_bytes(whisper), info


def add_bytes(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = int(time.time())
    tar.addfile(info, io.BytesIO(data))


def write_file(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def pick_val(rows, n, rng):
    """Pick up to n clips, cycling through speakers so each contributes about equally."""
    by_speaker = defaultdict(list)
    for r in rows:
        by_speaker[r["client_id"]].append(r)
    for clips in by_speaker.values():
        rng.shuffle(clips)
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
    picked = []
    while len(picked) < n and any(by_speaker.values()):
        for s in speakers:
            if by_speaker[s] and len(picked) < n:
                picked.append(by_speaker[s].pop())
    return picked


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("corpus", help="Common Voice folder with clips/ and the TSVs (only read)")
    parser.add_argument("n2w_clips", help="folder of Normal2Whisper <clip>.wav files (only read)")
    parser.add_argument("out_tar", help="tar file to write")
    parser.add_argument("--export-dir", help="also write the processed files here, for listening")
    parser.add_argument("--val-clips", type=int, default=500, help="validation clips from test-set speakers")
    parser.add_argument("--limit", type=int, help="only use this many train clips (for quick tests)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    corpus = os.path.realpath(args.corpus)
    n2w = os.path.realpath(args.n2w_clips)
    out = os.path.realpath(args.out_tar)
    export = os.path.realpath(args.export_dir) if args.export_dir else None
    for dst in filter(None, (out, export)):
        for src in (corpus, n2w):
            if (dst + os.sep).startswith(src + os.sep):
                sys.exit(f"{dst} must not be inside {src}")

    rng = random.Random(args.seed)
    validated = read_tsv(os.path.join(corpus, "validated.tsv"))
    test_speakers = {r["client_id"] for r in read_tsv(os.path.join(corpus, "test.tsv"))}
    durations = {r["clip"]: int(r["duration[ms]"]) for r in read_tsv(os.path.join(corpus, "clip_durations.tsv"))}

    train = [r for r in validated if r["client_id"] not in test_speakers]
    val = pick_val([r for r in validated if r["client_id"] in test_speakers], args.val_clips, rng)
    if args.limit is not None:
        rng.shuffle(train)
        train = train[:args.limit]
    rows = [("train", r) for r in train] + [("val", r) for r in val]

    missing = [r["path"] for _, r in rows if not os.path.exists(os.path.join(n2w, r["path"][:-4] + ".wav"))]
    if missing:
        sys.exit(f"{len(missing)} clips have no Normal2Whisper file, e.g. {missing[:3]}")

    prep = {"sample_rate": SR, "target_dbfs": TARGET_DBFS, "headroom_db": HEADROOM_DB, "max_gain_db": MAX_GAIN_DB,
            "min_speech_dbfs": MIN_SPEECH_DBFS, "thud_max_ms": THUD_MAX_MS, "thud_gap_ms": THUD_GAP_MS}
    prep_bytes = (json.dumps(prep, indent=1) + "\n").encode()

    partial = out + ".partial"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    started = time.time()
    kept, skipped = [], 0
    with tarfile.open(partial, "w") as tar:
        jobs = [(os.path.join(corpus, "clips", r["path"]), os.path.join(n2w, r["path"][:-4] + ".wav"))
                for _, r in rows]
        with ThreadPoolExecutor(args.jobs) as pool:
            for i, ((split, r), result) in enumerate(zip(rows, pool.map(prepare_pair, jobs)), 1):
                if result is None:
                    skipped += 1
                    continue
                normal, whisper, info = result
                name = r["path"][:-4]
                for kind, data in (("normal", normal), ("whisper", whisper)):
                    add_bytes(tar, f"{kind}/{name}.flac", data)
                    if export:
                        write_file(os.path.join(export, kind, name + ".flac"), data)
                kept.append((split, r, info))
                if i % 2000 == 0:
                    print(f"{i}/{len(rows)} pairs, {time.time() - started:.0f} s", flush=True)

        lines = ["clip\tsplit\tspeaker\tduration_ms\tnormal_dbfs\twhisper_dbfs\tthud_ms\tsentence"]
        for split, r, info in kept:
            sentence = " ".join(r["sentence"].split())  # no tabs or newlines in the TSV
            lines.append(f"{r['path'][:-4]}\t{split}\t{r['client_id']}\t{durations[r['path']]}\t"
                         f"{info['normal_dbfs']:.1f}\t{info['whisper_dbfs']:.1f}\t{info['thud_ms']:.0f}\t{sentence}")
        manifest = ("\n".join(lines) + "\n").encode()
        add_bytes(tar, "manifest.tsv", manifest)
        add_bytes(tar, "prep.json", prep_bytes)
        if export:
            write_file(os.path.join(export, "manifest.tsv"), manifest)
            write_file(os.path.join(export, "prep.json"), prep_bytes)
    os.replace(partial, out)

    print(f"wrote {out}: {len(kept)} pairs, {os.path.getsize(out) / 1e9:.2f} GB, {time.time() - started:.0f} s")
    for split in ("train", "val"):
        clips = [(r, info) for s, r, info in kept if s == split]
        hours = sum(durations[r["path"]] for r, _ in clips) / 3.6e6
        print(f"  {split}: {len(clips)} clips, {hours:.1f} h, {len({r['client_id'] for r, _ in clips})} speakers")
    levels = np.array([info["normal_dbfs"] for _, _, info in kept])
    thuds = np.array([info["thud_ms"] for _, _, info in kept])
    clipped = np.array([info["clipped"] for _, _, info in kept])
    print(f"  skipped {skipped} near-silent clips (speech below {MIN_SPEECH_DBFS} dBFS)")
    print("  speech level before normalization (dBFS): " + ", ".join(
        f"p{p} {np.percentile(levels, p):.1f}" for p in (1, 5, 50, 95, 99)))
    print(f"  thud muted in {(thuds > 0).sum()} clips ({(thuds > 0).mean() * 100:.0f}%), "
          f"median {np.median(thuds[thuds > 0]) if (thuds > 0).any() else 0:.0f} ms")
    print(f"  clips with any clipped samples: {(clipped > 0).sum()}, most in one clip: {clipped.max()}")


if __name__ == "__main__":
    main()
