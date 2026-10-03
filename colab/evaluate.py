"""Evaluate a fine-tuned WESPER encoder against the original on held-out Swedish clips.

Two stages, because they need different Python environments.

1. convert (this repo's .venv, which runs WESPER's code). For every held-out clip it measures
   how far each encoder's units are from the original encoder's units for the normal recording
   (the training target), and writes these recordings and conversions to OUT_DIR/<condition>/:

     normal-rec    the normal recording, normalized
     whisper-rec   its Normal2Whisper version, normalized
     normal-orig   normal speech through the original encoder: the best this decoder can do
     whisper-orig  the whisper through the original encoder: the baseline
     whisper-ft    the whisper through the fine-tuned encoder
     normal-ft     normal speech through the fine-tuned encoder

2. asr (a venv with transformers). It transcribes every condition with a Swedish speech
   recognizer (KB-Whisper) and reports character and word error rates against the sentences,
   plus a paired comparison of whisper-ft against whisper-orig.

Held-out clips are the val clips that the training run did not use to pick encoder_best.pt.
The notebook validates on the first VAL_CLIPS of them (default 200), so --skip defaults to 200.
All val speakers are outside the training set.

usage:
  .venv/bin/python colab/evaluate.py convert DATA_DIR ENCODER OUT_DIR [--skip 200] [--limit N]
  colab/.venv-eval/bin/python colab/evaluate.py asr OUT_DIR [--model KBLab/kb-whisper-small]

DATA_DIR is prepare_data.py's --export-dir (manifest.tsv, prep.json, normal/, whisper/).
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import time

import numpy as np
import soundfile as sf

SR = 16000
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE = "https://github.com/rkmt/wesper-demo/releases/download/v0.1"
ORIGINAL_ENCODER = f"{RELEASE}/model-layer12-450000.pt"
CONDITIONS = ["normal-rec", "whisper-rec", "normal-orig", "whisper-orig", "whisper-ft", "normal-ft"]


def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def write_tsv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fields, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def heldout_rows(manifest, skip):
    """Val clips after the first `skip`, which the training run used to pick its best checkpoint."""
    return [r for r in manifest if r["split"] == "val"][skip:]


# ---- convert ----------------------------------------------------------------------------------

def convert(args):
    import torch
    sys.path.insert(0, REPO_DIR)
    os.chdir(REPO_DIR)  # WESPER loads its configs by relative paths
    import whisper_normal as wn

    with open(os.path.join(args.data, "prep.json")) as f:
        headroom = 10 ** (json.load(f)["headroom_db"] / 20)
    rows = heldout_rows(read_tsv(os.path.join(args.data, "manifest.tsv")), args.skip)[: args.limit]
    if not rows:
        sys.exit("no held-out clips: check --skip against the number of val clips")

    w2n = wn.MyWhisper2Normal(argparse.Namespace(
        preprocess_config="config/my_preprocess16k_LJ.yaml", model_config="config/my_model16000.yaml",
        device="cpu", hubert=None, fastspeech2=f"{RELEASE}/googletts_neutral_best.tar",
        hifigan=f"{RELEASE}/g_00205000"), load_encoder=False)
    encoders = {"orig": wn.load_hubert(ORIGINAL_ENCODER, device="cpu"),
                "ft": wn.load_hubert(args.encoder, device="cpu")}

    def units(encoder, x):
        return encoder.units(torch.from_numpy(x)[None, None])[0]

    for cond in CONDITIONS:
        os.makedirs(os.path.join(args.out, cond), exist_ok=True)
    distances, started = [], time.time()
    for i, r in enumerate(rows, 1):
        audio = {}
        for kind in ("normal", "whisper"):
            x, _ = sf.read(os.path.join(args.data, kind, r["clip"] + ".flac"), dtype="float32")
            audio[kind] = x * headroom  # back to the training level
            # Float WAV: normalized peaks can exceed 1.0.
            sf.write(os.path.join(args.out, f"{kind}-rec", r["clip"] + ".wav"), audio[kind], SR, subtype="FLOAT")
        target = units(encoders["orig"], audio["normal"])
        row = {"clip": r["clip"]}
        for kind, enc in (("whisper", "orig"), ("whisper", "ft"), ("normal", "ft")):
            row[f"{kind}_{enc}"] = f"{(units(encoders[enc], audio[kind]) - target).abs().mean().item():.5f}"
        distances.append(row)
        for kind, enc in (("normal", "orig"), ("whisper", "orig"), ("whisper", "ft"), ("normal", "ft")):
            w2n.encoder = encoders[enc]
            out, _ = w2n.convert(audio[kind])
            sf.write(os.path.join(args.out, f"{kind}-{enc}", r["clip"] + ".wav"), out, SR, subtype="PCM_16")
        if i % 25 == 0 or i == len(rows):
            print(f"{i}/{len(rows)} clips, {time.time() - started:.0f} s", flush=True)

    write_tsv(os.path.join(args.out, "units.tsv"), distances, ["clip", "whisper_orig", "whisper_ft", "normal_ft"])
    write_tsv(os.path.join(args.out, "reference.tsv"), rows, ["clip", "speaker", "sentence"])
    d = {k: np.array([float(row[k]) for row in distances]) for k in ("whisper_orig", "whisper_ft", "normal_ft")}
    print(f"unit distance to the normal recording's original units, over {len(rows)} held-out clips:")
    print(f"  whisper, original encoder:   {d['whisper_orig'].mean():.4f}")
    print(f"  whisper, fine-tuned encoder: {d['whisper_ft'].mean():.4f} "
          f"(closer on {(d['whisper_ft'] < d['whisper_orig']).mean() * 100:.0f}% of clips)")
    print(f"  normal,  fine-tuned encoder: {d['normal_ft'].mean():.4f} (original: 0 by definition)")


# ---- asr --------------------------------------------------------------------------------------

def normalize_text(text):
    """Lowercase, letters and digits only, single spaces: so punctuation and case don't count as errors."""
    return " ".join(re.sub(r"[^\w]+|_", " ", text.lower()).split())


def edit_distance(ref, hyp):
    """Levenshtein distance between two sequences (strings or word lists)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def score(ref, hyp):
    """Character and word edits, and reference lengths, after normalizing both texts."""
    ref, hyp = normalize_text(ref), normalize_text(hyp)
    return {"char_edits": edit_distance(ref, hyp), "chars": len(ref),
            "word_edits": edit_distance(ref.split(), hyp.split()), "words": len(ref.split())}


def error_rate(scores, unit="char"):
    """Micro-averaged: all edits over all reference characters (or words)."""
    total = sum(s[f"{unit}s"] for s in scores)
    return sum(s[f"{unit}_edits"] for s in scores) / total if total else 0.0


def paired_bootstrap(a, b, unit="char", resamples=2000, seed=0):
    """95% interval for error_rate(b) - error_rate(a), resampling clips (a and b are paired by clip)."""
    rng = random.Random(seed)
    n, diffs = len(a), []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(error_rate([b[i] for i in idx], unit) - error_rate([a[i] for i in idx], unit))
    diffs.sort()
    return diffs[int(0.025 * resamples)], diffs[int(0.975 * resamples) - 1]


def kb_whisper(model_id):
    """Return transcribe(list of float arrays at 16 kHz) -> list of texts."""
    import torch
    from transformers import pipeline
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    asr = pipeline("automatic-speech-recognition", model=model_id, device=device)

    def transcribe(waves):
        # Common Voice sentences need well under 128 tokens. The cap stops a hallucinating
        # transcript from holding up its whole batch until Whisper's 448-token limit.
        # One clip at a time: measured ~3x faster than batches of 8 on Apple GPUs (MPS).
        outs = asr([{"raw": w, "sampling_rate": SR} for w in waves], batch_size=1,
                   generate_kwargs={"language": "sv", "task": "transcribe", "max_new_tokens": 128})
        return [o["text"] for o in outs]
    return transcribe


def transcribe_condition(out_dir, cond, refs, transcribe, chunk=64):
    """Transcripts for one condition, cached in asr-cache/<cond>.tsv so a rerun skips finished ones."""
    cache = os.path.join(out_dir, "asr-cache", cond + ".tsv")
    if os.path.exists(cache):
        cached = {r["clip"]: r["hypothesis"] for r in read_tsv(cache)}
        if all(r["clip"] in cached for r in refs):
            return [cached[r["clip"]] for r in refs]
    hyps, started = [], time.time()
    for i in range(0, len(refs), chunk):
        waves = [sf.read(os.path.join(out_dir, cond, r["clip"] + ".wav"), dtype="float32")[0] for r in refs[i:i + chunk]]
        hyps += [" ".join(h.split()) for h in transcribe(waves)]
        print(f"  {cond}: {len(hyps)}/{len(refs)} clips, {time.time() - started:.0f} s", flush=True)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    write_tsv(cache, [{"clip": r["clip"], "hypothesis": h} for r, h in zip(refs, hyps)], ["clip", "hypothesis"])
    return hyps


def run_asr(out_dir, transcribe):
    """Transcribe every condition in out_dir, write asr.tsv and summary.md, and return the summary text."""
    refs = read_tsv(os.path.join(out_dir, "reference.tsv"))
    conditions = [c for c in CONDITIONS if os.path.isdir(os.path.join(out_dir, c))]
    results, scores = [], {}
    for cond in conditions:
        hyps = transcribe_condition(out_dir, cond, refs, transcribe)
        scores[cond] = [score(r["sentence"], h) for r, h in zip(refs, hyps)]
        results += [{"clip": r["clip"], "condition": cond, "hypothesis": h, **s}
                    for r, h, s in zip(refs, hyps, scores[cond])]
        print(f"{cond}: CER {error_rate(scores[cond]) * 100:.1f}%", flush=True)
    write_tsv(os.path.join(out_dir, "asr.tsv"), results,
              ["clip", "condition", "char_edits", "chars", "word_edits", "words", "hypothesis"])

    lines = [f"Held-out clips: {len(refs)}, from {len({r['speaker'] for r in refs})} speakers not in training.", "",
             "| Condition | CER | WER |", "|---|---|---|"]
    lines += [f"| {c} | {error_rate(scores[c]) * 100:.1f}% | {error_rate(scores[c], 'word') * 100:.1f}% |"
              for c in conditions]
    if "whisper-orig" in scores and "whisper-ft" in scores:
        a, b = scores["whisper-orig"], scores["whisper-ft"]
        lo, hi = paired_bootstrap(a, b)
        delta = error_rate(b) - error_rate(a)
        better = sum(y["char_edits"] / max(y["chars"], 1) < x["char_edits"] / max(x["chars"], 1) for x, y in zip(a, b))
        worse = sum(y["char_edits"] / max(y["chars"], 1) > x["char_edits"] / max(x["chars"], 1) for x, y in zip(a, b))
        lines += ["", f"whisper-ft vs whisper-orig: CER {delta * 100:+.1f} points "
                      f"(95% interval {lo * 100:+.1f} to {hi * 100:+.1f}); "
                      f"better on {better} clips, worse on {worse}, same on {len(a) - better - worse}."]
    summary = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("convert", help="measure unit distances and write conversions")
    p.add_argument("data", help="prepare_data.py export folder")
    p.add_argument("encoder", help="fine-tuned encoder, e.g. encoder_best.pt")
    p.add_argument("out", help="output folder")
    p.add_argument("--skip", type=int, default=200, help="val clips the training run used for model selection")
    p.add_argument("--limit", type=int, help="only evaluate this many clips")
    p = sub.add_parser("asr", help="transcribe the conversions and score them")
    p.add_argument("out", help="output folder of the convert stage")
    p.add_argument("--model", default="KBLab/kb-whisper-small", help="Hugging Face speech recognition model")
    args = parser.parse_args()

    if args.stage == "convert":
        args.data, args.encoder, args.out = (os.path.realpath(p) for p in (args.data, args.encoder, args.out))
        convert(args)
    else:
        print(run_asr(os.path.realpath(args.out), kb_whisper(args.model)))


if __name__ == "__main__":
    main()
