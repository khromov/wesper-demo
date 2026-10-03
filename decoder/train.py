"""Train a WESPER decoder on one speaker, using data from decoder/prepare_data.py.

The decoder is WESPER's FastSpeech2 (libs/FastSpeech2), trained the way WESPER's own decoders
were: speech units in, mel spectrogram out, durations fixed at 1 unit = 1 frame, pitch and
energy predicted per frame. It uses the repo's loss (FastSpeech2Loss) and learning-rate schedule
(ScheduledOptim, settings from config/my_train16k_LJ.yaml). By default it starts from WESPER's
released Google TTS decoder (--init googletts), which needs far fewer steps than starting from
scratch (--init none). The new speaker's pitch and energy statistics replace the old ones.

RUN_DIR gets:
  decoder_best.pt     the best checkpoint so far (lowest validation mel loss), for WESPER:
                        --fastspeech2 RUN_DIR/decoder_best.pt --preprocess_config RUN_DIR/preprocess.yaml
  preprocess.yaml     WESPER's preprocessing config, pointing at this folder's stats.json
  stats.json          the speaker's pitch and energy statistics
  latest.pt           everything needed to resume; re-running the same command continues
  history.json        losses
  samples/            validation utterances: reference/ (the recording), vocoded-target/ (the
                      recording's own mel through the vocoder, the best possible result), and
                      step_NNNNNN/ (the decoder's output from the units, as WESPER produces it)

usage: python decoder/train.py DATA_DIR RUN_DIR [--steps 30000] [--batch-size 16] [--init googletts]
"""
import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time

import numpy as np
import soundfile as sf
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "libs", "FastSpeech2"))

RELEASE = "https://github.com/rkmt/wesper-demo/releases/download/v0.1"
INITS = {"googletts": f"{RELEASE}/googletts_neutral_best.tar", "lj": f"{RELEASE}/lambda_best.tar"}
VOCODER = f"{RELEASE}/g_00205000"
BINS = ("variance_adaptor.pitch_bins", "variance_adaptor.energy_bins")  # derived from stats.json, never loaded


def read_yaml(path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def fetch(path_or_url):
    """Local path of a checkpoint, downloading URLs into torch's hub cache once."""
    if not path_or_url.startswith("http"):
        return path_or_url
    cached = os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(path_or_url))
    if not os.path.exists(cached):
        os.makedirs(os.path.dirname(cached), exist_ok=True)
        torch.hub.download_url_to_file(path_or_url, cached)
    return cached


def configs(run_dir):
    """WESPER's configs, with the preprocessing config pointed at run_dir's stats.json."""
    pre = read_yaml(os.path.join(REPO, "config", "my_preprocess16k_LJ.yaml"))
    pre["dataset"] = os.path.basename(run_dir)
    # Relative to the repo, where WESPER runs from, so the run folder can move between computers.
    pre["path"] = {"preprocessed_path": os.path.relpath(run_dir, REPO)}
    return pre, read_yaml(os.path.join(REPO, "config", "my_model16000.yaml")), read_yaml(
        os.path.join(REPO, "config", "my_train16k_LJ.yaml"))


def read_segments(data_dir, max_frames):
    with open(os.path.join(data_dir, "segments.tsv"), newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [r for r in rows if 0 < int(r["frames"]) <= max_frames]


class Utterances(torch.utils.data.Dataset):
    """Units, mel, normalized pitch and energy of each utterance, all T frames long."""

    def __init__(self, data_dir, rows, stats):
        self.data_dir, self.rows = data_dir, rows
        self.pitch_norm, self.energy_norm = stats["pitch"][2:], stats["energy"][2:]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with np.load(os.path.join(self.data_dir, "segments", self.rows[i]["id"] + ".npz")) as d:
            return {"id": self.rows[i]["id"], "units": d["units"].astype(np.float32), "mel": d["mel"].T.astype(np.float32),
                    "pitch": ((d["pitch"] - self.pitch_norm[0]) / self.pitch_norm[1]).astype(np.float32),
                    "energy": ((d["energy"] - self.energy_norm[0]) / self.energy_norm[1]).astype(np.float32)}


def collate(items):
    """A padded batch in FastSpeech2's 12-field layout; model(*batch[2:]), loss(batch, output)."""
    lens = torch.tensor([len(it["units"]) for it in items])
    t = int(lens.max())

    def pad(key):
        out = torch.zeros((len(items), t) + items[0][key].shape[1:])
        for i, it in enumerate(items):
            out[i, : len(it[key])] = torch.from_numpy(it[key])
        return out
    durations = (torch.arange(t)[None, :] < lens[:, None]).long()  # 1 unit = 1 frame
    ids = [it["id"] for it in items]
    return (ids, ids, torch.zeros(len(items), dtype=torch.long), pad("units"), lens, t,
            pad("mel"), lens, t, pad("pitch"), pad("energy"), durations)


def length_batches(lengths, batch_size, rng):
    """Batches of similar-length utterances (less padding), in random order."""
    order = list(range(len(lengths)))
    rng.shuffle(order)
    batches = []
    for i in range(0, len(order), batch_size * 20):
        chunk = sorted(order[i:i + batch_size * 20], key=lambda k: lengths[k])
        batches += [chunk[j:j + batch_size] for j in range(0, len(chunk), batch_size)]
    rng.shuffle(batches)
    return [b for b in batches if len(b) == batch_size] or batches


def to_device(batch, device):
    return tuple(x.to(device, non_blocking=True) if torch.is_tensor(x) else x for x in batch)


def as_float(output):
    # Losses in float32, also when the forward pass ran in bfloat16 or float16.
    return tuple(o.float() if torch.is_tensor(o) and o.is_floating_point() else o for o in output)


def save(obj, path):
    torch.save(obj, path + ".tmp")  # write then rename: an interruption can't leave a broken file
    os.replace(path + ".tmp", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", help="output of decoder/prepare_data.py")
    parser.add_argument("run_dir", help="where checkpoints and samples go; re-run to resume")
    # 15 h of audio is ~350 steps per pass at batch 16, so 30,000 is ~85 passes: plenty for
    # fine-tuning. Re-running with a higher --steps continues where it stopped.
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--init", default="googletts", help="googletts, lj, none, or a checkpoint path")
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=6, help="validation utterances to synthesize at each eval")
    parser.add_argument("--device", default="auto", help="auto (cuda if available, else cpu), cuda, mps or cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_dir, run_dir = os.path.realpath(args.data_dir), os.path.realpath(args.run_dir)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)
    os.chdir(REPO)  # WESPER's code reads its configs by relative paths

    import utils.tools
    import model.modules
    from model import FastSpeech2, FastSpeech2Loss, ScheduledOptim
    from whisper_normal import load_hifigan
    utils.tools.device = model.modules.device = device  # module-level globals in WESPER's FastSpeech2

    shutil.copy(os.path.join(data_dir, "stats.json"), os.path.join(run_dir, "stats.json"))
    pre, model_config, train_config = configs(run_dir)
    with open(os.path.join(run_dir, "preprocess.yaml"), "w") as f:
        yaml.safe_dump(pre, f, sort_keys=False)
    with open(os.path.join(run_dir, "stats.json")) as f:
        stats = json.load(f)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_segments(data_dir, model_config["max_seq_len"])
    train_rows, val_rows = [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "val"]
    print(f"train: {len(train_rows)} utterances, val: {len(val_rows)}, device: {device}", flush=True)
    train_set, val_set = Utterances(data_dir, train_rows, stats), Utterances(data_dir, val_rows, stats)

    net = FastSpeech2(pre, model_config).to(device)
    loss_fn = FastSpeech2Loss(pre, model_config).to(device)
    latest = os.path.join(run_dir, "latest.pt")
    state = torch.load(latest, map_location="cpu", weights_only=False) if os.path.exists(latest) else None
    if state is not None:
        net.load_state_dict(state["model"])
        step, best, history = state["step"], state["best"], state["history"]
        print(f"resuming from step {step}", flush=True)
    else:
        step, best, history = 0, math.inf, {"train": [], "val": []}
        if args.init != "none":
            init = torch.load(fetch(INITS.get(args.init, args.init)), map_location="cpu", weights_only=False)["model"]
            missing, unexpected = net.load_state_dict({k: v for k, v in init.items() if k not in BINS}, strict=False)
            assert set(missing) == set(BINS) and not unexpected, (missing, unexpected)
            print(f"starting from {args.init}", flush=True)
    optimizer = ScheduledOptim(net, train_config, model_config, step)
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])

    amp = None
    if device == "cuda":
        amp = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp == torch.float16)
    if state is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    autocast = (lambda: torch.autocast("cuda", dtype=amp)) if amp else (lambda: torch.autocast("cpu", enabled=False))

    vocoder = load_hifigan(model_config, checkpoint_path=VOCODER, device=device)

    def synthesize(mel):
        with torch.no_grad():
            return vocoder(mel.to(device)[None]).squeeze().float().cpu().numpy()

    sample_rows = val_rows[: args.samples]
    if sample_rows and not os.path.exists(os.path.join(run_dir, "samples", "vocoded-target")):
        for kind in ("reference", "vocoded-target"):
            os.makedirs(os.path.join(run_dir, "samples", kind), exist_ok=True)
        for r, item in ((r, val_set[i]) for i, r in enumerate(sample_rows)):
            wav, _ = sf.read(os.path.join(data_dir, "audio", r["id"] + ".flac"), dtype="float32")
            sf.write(os.path.join(run_dir, "samples", "reference", r["id"] + ".wav"),
                     wav * 10 ** (float(r["gain_db"]) / 20), 16000, subtype="FLOAT")
            sf.write(os.path.join(run_dir, "samples", "vocoded-target", r["id"] + ".wav"),
                     synthesize(torch.from_numpy(item["mel"]).T), 16000, subtype="FLOAT")

    def evaluate():
        net.eval()
        totals, n = np.zeros(6), 0
        loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate)
        with torch.no_grad():
            for batch in loader:
                batch = to_device(batch, device)
                with autocast():
                    output = net(*batch[2:])
                totals += np.array([l.item() for l in loss_fn(batch, as_float(output))]) * len(batch[0])
                n += len(batch[0])
            folder = os.path.join(run_dir, "samples", f"step_{step:06d}")
            os.makedirs(folder, exist_ok=True)
            for i, r in enumerate(sample_rows):
                u = torch.from_numpy(val_set[i]["units"])[None].to(device)
                # As WESPER runs it: units only, durations and pitch predicted.
                out = net(torch.zeros(1, dtype=torch.long, device=device), u, torch.tensor([u.shape[1]], device=device), u.shape[1])
                sf.write(os.path.join(folder, r["id"] + ".wav"), synthesize(out[1][0].float().T), 16000, subtype="FLOAT")
        net.train()
        return dict(zip(["total", "mel", "postnet_mel", "pitch", "energy", "duration"], (totals / max(n, 1)).tolist()))

    rng = random.Random(args.seed + step)
    lengths = [int(r["frames"]) for r in train_rows]
    net.train()
    running, started, start_step = [], time.time(), step
    while step < args.steps:
        loader = torch.utils.data.DataLoader(
            train_set, batch_sampler=length_batches(lengths, args.batch_size, rng), collate_fn=collate,
            num_workers=args.workers, pin_memory=device == "cuda")
        for batch in loader:
            if step >= args.steps:
                break
            batch = to_device(batch, device)
            with autocast():
                output = net(*batch[2:])
            losses = loss_fn(batch, as_float(output))
            optimizer.zero_grad()
            scaler.scale(losses[0]).backward()
            scaler.unscale_(optimizer._optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), train_config["optimizer"]["grad_clip_thresh"])
            optimizer._update_learning_rate()  # ScheduledOptim.step_and_update_lr, split around the scaler
            scaler.step(optimizer._optimizer)
            scaler.update()
            step += 1
            running.append([l.item() for l in losses])

            if step % 100 == 0 or step == args.steps:
                m = np.mean(running, axis=0)
                rate = (step - start_step) / (time.time() - started)
                history["train"].append({"step": step, "total": m[0], "mel": m[1], "postnet_mel": m[2]})
                print(f"step {step}/{args.steps}  loss {m[0]:.3f} (mel {m[1]:.3f}, postnet {m[2]:.3f}, pitch {m[3]:.3f}, "
                      f"energy {m[4]:.3f})  lr {optimizer._optimizer.param_groups[0]['lr']:.1e}  "
                      f"{rate:.2f} steps/s  ETA {(args.steps - step) / rate / 60:.0f} min", flush=True)
                running = []
            if step % args.eval_every == 0 or step == args.steps:
                scores = evaluate()
                history["val"].append({"step": step, **scores})
                improved = scores["postnet_mel"] < best
                if improved:
                    best = scores["postnet_mel"]
                    save({"model": net.state_dict(), "step": step, "val": scores}, os.path.join(run_dir, "decoder_best.pt"))
                print(f"  val: mel {scores['mel']:.3f}, postnet {scores['postnet_mel']:.3f}, pitch {scores['pitch']:.3f}"
                      f"{'  -> saved decoder_best.pt' if improved else ''}", flush=True)
            if step % args.save_every == 0 or step == args.steps:
                save({"model": net.state_dict(), "optimizer": optimizer._optimizer.state_dict(), "scaler": scaler.state_dict(),
                      "step": step, "best": best, "history": history, "args": vars(args)}, latest)
                with open(os.path.join(run_dir, "history.json"), "w") as f:
                    json.dump(history, f, indent=1)

    rel = os.path.relpath(run_dir, REPO)
    print(f"done at step {step}. Use it with WESPER:\n  --fastspeech2 {rel}/decoder_best.pt --preprocess_config {rel}/preprocess.yaml")


if __name__ == "__main__":
    main()
