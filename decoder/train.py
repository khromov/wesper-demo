"""Train a WESPER decoder on one speaker, using data from decoder/prepare_data.py.

The decoder is WESPER's FastSpeech2 (libs/FastSpeech2), trained the way WESPER's own decoders
were: speech units in, mel spectrogram out, durations fixed at 1 unit = 1 frame, pitch and
energy predicted per frame. It uses the repo's loss (FastSpeech2Loss) and learning-rate schedule
(ScheduledOptim, settings from config/my_train16k_LJ.yaml). By default it starts from WESPER's
released Google TTS decoder (--init googletts), which needs far fewer steps than starting from
scratch (--init none). The new speaker's pitch and energy statistics replace the old ones.

The decoder is trained for the vocoder its data was prepared for (prepare_data.py --vocoder,
recorded in DATA_DIR/prep.json; see vocoders.py). The run's preprocess.yaml records it too,
and WESPER loads that vocoder when it uses the decoder.

RUN_DIR gets:
  decoder_best.pt     the best checkpoint so far (lowest validation mel loss), for WESPER:
                        --fastspeech2 RUN_DIR/decoder_best.pt --preprocess_config RUN_DIR/preprocess.yaml
  preprocess.yaml     WESPER's preprocessing config, pointing at this folder's stats.json
  stats.json          the speaker's pitch and energy statistics
  latest.pt           everything needed to resume; re-running the same command continues
  history.json        losses
  samples/            validation utterances: reference/ (the recording, 16 kHz), vocoded-target/ (the
                      recording's own mel through the vocoder, the best possible result), and
                      step_NNNNNN/ (the decoder's output from the units, as WESPER produces it)

Each validation also measures the buzz BigVGAN makes from over-smoothed mels: the spectral
flatness of the samples in 0.5-2, 2-4 and 4-8 kHz (lower is less noise-like; compare with
vocoded-target's), and the mels' sharpness, the mean difference between adjacent mel bands
(compare with the real mels'). Both use the decoder as WESPER runs it: pitch predicted.

Fine-tuning a finished decoder (all opt-in):
  --init RUN/decoder_best.pt     start from a finished run's weights (any decoder checkpoint)
  --reset-postnet                give the postnet a fresh start; see reset_postnet()
  --lr 1e-4 [--warmup 200]       a fine-tuning learning rate: warmup, then a cosine decay to a tenth
  --train-only postnet           train only the postnet; the rest stays fixed (and without dropout)
  --keep-checkpoints             also keep each validation's weights in checkpoints/step_NNNNNN.pt
  --benchmark 50                 time 50 training steps, print the speed, and exit without saving

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
import vocoders  # noqa: E402

RELEASE = "https://github.com/rkmt/wesper-demo/releases/download/v0.1"
INITS = {"googletts": f"{RELEASE}/googletts_neutral_best.tar", "lj": f"{RELEASE}/lambda_best.tar"}
VOCODER = vocoders.HIFIGAN16K.checkpoint
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


def configs(run_dir, voc=vocoders.HIFIGAN16K):
    """WESPER's configs, with the preprocessing config pointed at run_dir's stats.json and set
    up for the vocoder (sample rate, STFT and mel settings, and its name for WESPER)."""
    pre = read_yaml(os.path.join(REPO, "config", "my_preprocess16k_LJ.yaml"))
    pre["dataset"] = os.path.basename(run_dir)
    # Relative to the repo, where WESPER runs from, so the run folder can move between computers.
    pre["path"] = {"preprocessed_path": os.path.relpath(run_dir, REPO)}
    p = pre["preprocessing"]
    p["audio"]["sampling_rate"] = voc.sample_rate
    p["stft"].update(filter_length=voc.n_fft, hop_length=voc.hop, win_length=voc.win)
    p["mel"].update(n_mel_channels=voc.n_mels, mel_fmin=voc.fmin, mel_fmax=voc.fmax)
    pre["vocoder"] = {"name": voc.name}
    return pre, read_yaml(os.path.join(REPO, "config", "my_model16000.yaml")), read_yaml(
        os.path.join(REPO, "config", "my_train16k_LJ.yaml"))


def load_init(name_or_path):
    """The weights to start from: a release (googletts, lj) or any decoder checkpoint: decoder_best.pt,
    checkpoints/step_NNNNNN.pt, latest.pt, or a bare state dict."""
    state = torch.load(fetch(INITS.get(name_or_path, name_or_path)), map_location="cpu", weights_only=False)
    return state["model"] if "model" in state else state


def reset_postnet(net, mode):
    """Gives the postnet a fresh start. Its last BatchNorm's scale is ~0.01 in every trained decoder
    (0.0015 in the googletts release), against ~4 for the others, so it adds next to nothing to the
    mel. "final-norm" resets that BatchNorm only; "all" re-initializes the whole postnet."""
    from transformer import PostNet
    final = net.postnet.convolutions[-1][1]
    if mode == "all":
        fresh = PostNet(n_mel_channels=final.num_features).to(final.weight.device)
        net.postnet.load_state_dict(fresh.state_dict())
    else:
        final.reset_parameters()  # scale 1, shift 0, running mean 0 and variance 1


def finetune_lr(step, lr, warmup, steps):
    """--lr's schedule: a linear warmup to lr over `warmup` steps, then a cosine decay to lr / 10 at `steps`."""
    if step < warmup:
        return lr * (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, steps - warmup))
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * t)))


def train_mode(net, only=None):
    """net.train(); with --train-only postnet, the fixed part stays in eval mode, so the postnet learns
    to correct the mels the decoder makes at inference (without dropout)."""
    net.train()
    if only == "postnet":
        net.eval()
        net.postnet.train()


FLATNESS_BANDS = ((500, 2000), (2000, 4000), (4000, 8000))  # Hz


def band_flatness(wav, sample_rate, n_fft=1024, hop=256):
    """Spectral flatness in each of FLATNESS_BANDS: the geometric over the arithmetic mean of a frame's
    power spectrum, averaged over the loudest half of the frames. Near 1 for noise, low for a voice's
    harmonics; BigVGAN's buzz raises it."""
    x = torch.as_tensor(np.asarray(wav, dtype=np.float32))
    p = torch.stft(x, n_fft, hop, window=torch.hann_window(n_fft), return_complex=True).abs().pow(2).double().numpy() + 1e-12
    level = p.sum(0)
    loud = level >= np.median(level)
    freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    out = []
    for lo, hi in FLATNESS_BANDS:
        b = p[(freqs >= lo) & (freqs < hi)][:, loud]
        out.append(float(np.mean(np.exp(np.log(b).mean(0)) / b.mean(0))))
    return out


def mel_sharpness(mel):
    """The mean absolute difference between adjacent mel bands (mel: T x n_mels): how much spectral
    detail there is. The decoder's mels have less than real ones."""
    return float(np.abs(np.diff(np.asarray(mel, dtype=np.float64), axis=1)).mean())


def read_segments(data_dir, max_frames):
    with open(os.path.join(data_dir, "segments.tsv"), newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [r for r in rows if 0 < int(r["frames"]) <= max_frames]


class Utterances(torch.utils.data.Dataset):
    """Units, mel, normalized pitch and energy of each utterance, all T mel frames long.

    The units are stored one per 20 ms; for vocoders with other frame rates they're interpolated
    to the mel frames, exactly as WESPER does at inference (vocoders.units_to_frames).
    """

    def __init__(self, data_dir, rows, stats, voc=vocoders.HIFIGAN16K):
        self.data_dir, self.rows, self.voc = data_dir, rows, voc
        self.pitch_norm, self.energy_norm = stats["pitch"][2:], stats["energy"][2:]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with np.load(os.path.join(self.data_dir, "segments", self.rows[i]["id"] + ".npz")) as d:
            units = vocoders.units_to_frames(d["units"].astype(np.float32), d["mel"].shape[1], self.voc)
            return {"id": self.rows[i]["id"], "units": units, "mel": d["mel"].T.astype(np.float32),
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
    # Fine-tuning a finished decoder; see the module docstring.
    parser.add_argument("--reset-postnet", nargs="?", const="final-norm", choices=["final-norm", "all"],
                        help="reset the postnet's last BatchNorm (default), or all of the postnet")
    parser.add_argument("--lr", type=float, help="a fine-tuning learning rate instead of FastSpeech2's schedule (which peaks at 1e-3)")
    parser.add_argument("--warmup", type=int, default=200, help="steps to reach --lr")
    parser.add_argument("--postnet-final-dropout", type=float,
                        help="dropout on the postnet's correction in training (FastSpeech2: 0.5)")
    parser.add_argument("--train-only", choices=["postnet"], help="train only the postnet")
    parser.add_argument("--keep-checkpoints", action="store_true", help="keep each validation's weights in checkpoints/")
    parser.add_argument("--benchmark", type=int, metavar="N", help="time N training steps, print the speed, and exit without saving")
    args = parser.parse_args()

    data_dir, run_dir = os.path.realpath(args.data_dir), os.path.realpath(args.run_dir)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)
    os.chdir(REPO)  # WESPER's code reads its configs by relative paths

    import utils.tools
    import model.modules
    from model import FastSpeech2, FastSpeech2Loss, ScheduledOptim
    utils.tools.device = model.modules.device = device  # module-level globals in WESPER's FastSpeech2

    with open(os.path.join(data_dir, "prep.json")) as f:
        voc = vocoders.spec(json.load(f).get("vocoder", vocoders.DEFAULT))
    shutil.copy(os.path.join(data_dir, "stats.json"), os.path.join(run_dir, "stats.json"))
    pre, model_config, train_config = configs(run_dir, voc)
    with open(os.path.join(run_dir, "preprocess.yaml"), "w") as f:
        yaml.safe_dump(pre, f, sort_keys=False)
    with open(os.path.join(run_dir, "stats.json")) as f:
        stats = json.load(f)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_segments(data_dir, model_config["max_seq_len"])
    train_rows, val_rows = [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "val"]
    print(f"train: {len(train_rows)} utterances, val: {len(val_rows)}, vocoder: {voc.name}, device: {device}", flush=True)
    train_set, val_set = Utterances(data_dir, train_rows, stats, voc), Utterances(data_dir, val_rows, stats, voc)

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
            init = load_init(args.init)
            own = net.state_dict()
            # Layers whose shape differs (the mel output, for a vocoder with other than 80 bands) start fresh.
            reshaped = {k for k, v in init.items() if k in own and v.shape != own[k].shape}
            missing, unexpected = net.load_state_dict(
                {k: v for k, v in init.items() if k not in BINS and k not in reshaped}, strict=False)
            assert set(missing) == set(BINS) | reshaped and not unexpected, (missing, unexpected)
            print(f"starting from {args.init}" + (f", except {sorted(reshaped)}" if reshaped else ""), flush=True)
        if args.reset_postnet:
            reset_postnet(net, args.reset_postnet)
            print(f"reset the postnet ({args.reset_postnet})", flush=True)
    if args.postnet_final_dropout is not None:
        net.postnet.final_dropout = args.postnet_final_dropout
    if args.train_only == "postnet":
        for name, p in net.named_parameters():
            p.requires_grad = name.startswith("postnet.")
        print(f"training only the postnet: {sum(p.numel() for p in net.postnet.parameters()):,} parameters", flush=True)
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

    vocoder = vocoders.load(voc, device)

    def synthesize(mel):
        return vocoders.synthesize(vocoder, mel)

    sample_rows = val_rows[: args.samples] if not args.benchmark else []
    if sample_rows and not os.path.exists(os.path.join(run_dir, "samples", "vocoded-target")):
        for kind in ("reference", "vocoded-target"):
            os.makedirs(os.path.join(run_dir, "samples", kind), exist_ok=True)
        for r, item in ((r, val_set[i]) for i, r in enumerate(sample_rows)):
            wav, _ = sf.read(os.path.join(data_dir, "audio", r["id"] + ".flac"), dtype="float32")
            sf.write(os.path.join(run_dir, "samples", "reference", r["id"] + ".wav"),
                     wav * 10 ** (float(r["gain_db"]) / 20), 16000, subtype="FLOAT")
            sf.write(os.path.join(run_dir, "samples", "vocoded-target", r["id"] + ".wav"),
                     synthesize(torch.from_numpy(item["mel"]).T), voc.sample_rate, subtype="FLOAT")

    # The buzz measures' reference: the flatness of the recordings' own mels through the vocoder.
    target_flatness = [band_flatness(sf.read(os.path.join(run_dir, "samples", "vocoded-target", r["id"] + ".wav"),
                                             dtype="float32")[0], voc.sample_rate) for r in sample_rows]
    target_flatness = np.mean(target_flatness, axis=0).tolist() if target_flatness else None

    def evaluate():
        net.eval()
        totals, n = np.zeros(6), 0
        # with the pitch predicted, as WESPER runs it: the mel error and the mels' sharpness, theirs and the real ones'
        inference, sharp, sharp_real = [], [], []
        loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate)
        with torch.no_grad():
            for batch in loader:
                batch = to_device(batch, device)
                with autocast():
                    output = net(*batch[2:])
                    predicted = net(*batch[2:6])
                totals += np.array([l.item() for l in loss_fn(batch, as_float(output))]) * len(batch[0])
                n += len(batch[0])
                for k in range(len(batch[0])):
                    real = batch[6][k, : int(batch[7][k])].float().cpu().numpy()
                    mel = predicted[1][k, : int(predicted[9][k])].float().cpu().numpy()
                    t = min(len(mel), len(real))
                    inference.append(np.abs(mel[:t] - real[:t]).mean())
                    sharp.append(mel_sharpness(mel))
                    sharp_real.append(mel_sharpness(real))
            folder = os.path.join(run_dir, "samples", f"step_{step:06d}")
            os.makedirs(folder, exist_ok=True)
            flatness = []
            for i, r in enumerate(sample_rows):
                u = torch.from_numpy(val_set[i]["units"])[None].to(device)
                # As WESPER runs it: units only, durations and pitch predicted.
                out = net(torch.zeros(1, dtype=torch.long, device=device), u, torch.tensor([u.shape[1]], device=device), u.shape[1])
                wav = synthesize(out[1][0].float().T)
                flatness.append(band_flatness(wav, voc.sample_rate))
                sf.write(os.path.join(folder, r["id"] + ".wav"), wav, voc.sample_rate, subtype="FLOAT")
        train_mode(net, args.train_only)
        scores = dict(zip(["total", "mel", "postnet_mel", "pitch", "energy", "duration"], (totals / max(n, 1)).tolist()))
        scores.update(mel_inference=float(np.mean(inference)), sharpness=float(np.mean(sharp)),
                      sharpness_real=float(np.mean(sharp_real)))
        if flatness:
            scores["flatness"] = np.mean(flatness, axis=0).tolist()
        return scores

    rng = random.Random(args.seed + step)
    lengths = [int(r["frames"]) for r in train_rows]
    train_mode(net, args.train_only)
    running, started, start_step = [], time.time(), step
    timed = (step, started)  # --benchmark: timed from here, or after 3 warm-up steps
    while step < args.steps or args.benchmark:
        loader = torch.utils.data.DataLoader(
            train_set, batch_sampler=length_batches(lengths, args.batch_size, rng), collate_fn=collate,
            num_workers=args.workers, pin_memory=device == "cuda")
        for batch in loader:
            if step >= args.steps and not args.benchmark:
                break
            batch = to_device(batch, device)
            with autocast():
                output = net(*batch[2:])
            losses = loss_fn(batch, as_float(output))
            optimizer.zero_grad()
            scaler.scale(losses[0]).backward()
            scaler.unscale_(optimizer._optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), train_config["optimizer"]["grad_clip_thresh"])
            if args.lr:
                for group in optimizer._optimizer.param_groups:
                    group["lr"] = finetune_lr(step, args.lr, args.warmup, args.steps)
            else:
                optimizer._update_learning_rate()  # ScheduledOptim.step_and_update_lr, split around the scaler
            scaler.step(optimizer._optimizer)
            scaler.update()
            step += 1
            running.append([l.item() for l in losses])

            if args.benchmark:
                if device == "cuda":
                    torch.cuda.synchronize()
                if step - start_step == 3 and args.benchmark > 3:
                    timed = (step, time.time())
                if step - start_step >= args.benchmark:
                    per_step = (time.time() - timed[1]) / (step - timed[0])
                    left = max(args.steps - start_step, 0)
                    print(f"benchmark: {per_step:.2f} s/step ({1 / per_step:.2f} steps/s) over {step - timed[0]} steps at batch "
                          f"{args.batch_size}; {left} steps (--steps {args.steps}) would take {left * per_step / 3600:.1f} h, "
                          f"plus validations. Nothing was saved.", flush=True)
                    return
                continue

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
                if args.keep_checkpoints:
                    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
                    save({"model": net.state_dict(), "step": step, "val": scores},
                         os.path.join(run_dir, "checkpoints", f"step_{step:06d}.pt"))
                print(f"  val: mel {scores['mel']:.3f}, postnet {scores['postnet_mel']:.3f}, pitch {scores['pitch']:.3f}"
                      f"{'  -> saved decoder_best.pt' if improved else ''}", flush=True)
                flat = "/".join(f"{x:.3f}" for x in scores.get("flatness", []))
                target = "/".join(f"{x:.3f}" for x in target_flatness or [])
                print(f"  buzz (pitch predicted): flatness 0.5-2/2-4/4-8 kHz {flat or '-'} (vocoded target {target or '-'}), "
                      f"sharpness {scores['sharpness']:.3f} (real {scores['sharpness_real']:.3f}), "
                      f"mel {scores['mel_inference']:.3f}", flush=True)
            if step % args.save_every == 0 or step == args.steps:
                save({"model": net.state_dict(), "optimizer": optimizer._optimizer.state_dict(), "scaler": scaler.state_dict(),
                      "step": step, "best": best, "history": history, "args": vars(args)}, latest)
                with open(os.path.join(run_dir, "history.json"), "w") as f:
                    json.dump(history, f, indent=1)

    rel = os.path.relpath(run_dir, REPO)
    print(f"done at step {step}. Use it with WESPER:\n  --fastspeech2 {rel}/decoder_best.pt --preprocess_config {rel}/preprocess.yaml")


if __name__ == "__main__":
    main()
