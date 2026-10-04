"""Fine-tune a decoder run's BigVGAN vocoder on that decoder's own output.

The decoder's mel spectrograms are smoother than real ones (its L1 loss rewards the average), and
BigVGAN, trained only on real mels, turns the missing detail into a buzz. Fine-tuning teaches it
to make the recordings from the decoder's mels. The decoder makes one frame per frame of the
recording, so its output lines up with the recording frame by frame:

1. The decoder (DECODER_RUN/decoder_best.pt) computes every utterance's mel as in its training:
   from the units, with the recording's own pitch and energy, so that the mel also matches the
   recording's intonation. The mels are cached in --cache.
2. BigVGAN learns to turn random crops of them into the same stretch of the recording (AUDIO_DIR,
   from decoder/export_vocoder_audio.py), with NVIDIA's losses and discriminators. It starts from
   NVIDIA's released generator, discriminators and optimizer states (bigvgan_discriminator_optimizer.pt,
   1.4 GB), so the adversarial training picks up where theirs ended.

Validation measures how close the vocoder gets to the recordings from the decoder's mels: the mean
absolute difference between the mel of its output and the recording's mel ("val mel"), over the
validation utterances. Step 0 is the original vocoder.

OUT_DIR becomes a decoder run for WESPER: DECODER_RUN's decoder with the fine-tuned vocoder.
  bigvgan_generator.pt  the vocoder with the best val mel so far (at first the original), in NVIDIA's format
  decoder_best.pt, stats.json  copied from DECODER_RUN
  preprocess.yaml       DECODER_RUN's, pointing at this folder and at bigvgan_generator.pt
  latest.pt             everything needed to resume; re-running the same command continues
  history.json          losses
  samples/              validation utterances: reference/ (the recording), and the vocoder's output
                        at step_000000/ (the original vocoder) and at each validation, from the
                        decoder's mels as WESPER makes them (units only, pitch predicted)

Use it with WESPER like any decoder run: DECODER=OUT_DIR ./client_direct_sv.sh

usage: python decoder/finetune_vocoder.py DATA_DIR AUDIO_DIR DECODER_RUN OUT_DIR [--steps 20000] [--batch-size 8]
"""
import argparse
import itertools
import json
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
from decoder import export_vocoder_audio, train as decoder_train  # noqa: E402  (also puts FastSpeech2 on the path)
import vocoders  # noqa: E402

SAMPLES = 6


def save(obj, path):
    torch.save(obj, path + ".tmp")  # write then rename: an interruption can't leave a broken file
    os.replace(path + ".tmp", path)


def load_decoder(run_dir, device):
    """A decoder/train.py run's decoder_best.pt in eval mode, and FastSpeech2's model config."""
    import model.modules
    import utils.tools
    from model import FastSpeech2
    utils.tools.device = model.modules.device = device  # module-level globals in WESPER's FastSpeech2
    pre = decoder_train.read_yaml(os.path.join(run_dir, "preprocess.yaml"))
    pre["path"]["preprocessed_path"] = run_dir  # its stats.json, wherever the folder is
    model_config = decoder_train.read_yaml(os.path.join(REPO, "config", "my_model16000.yaml"))
    net = FastSpeech2(pre, model_config)
    state = torch.load(os.path.join(run_dir, "decoder_best.pt"), map_location="cpu", weights_only=False)
    net.load_state_dict(state["model"], strict=True)
    return net.to(device).eval(), model_config


@torch.no_grad()
def cache_mels(net, dataset, cache, device, batch_size=16):
    """The decoder's mel (n_mels x T, float16) of each utterance, with its own pitch and energy, as in training."""
    os.makedirs(cache, exist_ok=True)
    todo = [i for i, r in enumerate(dataset.rows) if not os.path.exists(os.path.join(cache, r["id"] + ".npy"))]
    for start in range(0, len(todo), batch_size):
        idx = todo[start:start + batch_size]
        batch = decoder_train.to_device(decoder_train.collate([dataset[i] for i in idx]), device)
        mels = net(*batch[2:])[1].float().cpu().numpy()  # postnet output, B x T x n_mels
        for k, i in enumerate(idx):
            path = os.path.join(cache, dataset.rows[i]["id"] + ".npy")
            with open(path + ".tmp", "wb") as f:
                np.save(f, mels[k, : int(batch[7][k])].T.astype(np.float16))
            os.replace(path + ".tmp", path)
    return len(todo)


@torch.no_grad()
def inference_mel(net, item, device):
    """The decoder's mel as WESPER makes it: from the units only, pitch and energy predicted (decoder/train.py's samples)."""
    u = torch.from_numpy(item["units"])[None].to(device)
    out = net(torch.zeros(1, dtype=torch.long, device=device), u, torch.tensor([u.shape[1]], device=device), u.shape[1])
    return out[1][0].float().T.contiguous()


class Crops(torch.utils.data.Dataset):
    """Random crops of `frames` of the decoder's mel frames, and the same stretch of the recording."""

    def __init__(self, ids, cache, audio_dir, frames, hop):
        self.ids, self.cache, self.audio_dir, self.frames, self.hop = ids, cache, audio_dir, frames, hop

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        mel = np.load(os.path.join(self.cache, self.ids[i] + ".npy"), mmap_mode="r")
        f = random.randrange(mel.shape[1] - self.frames + 1)
        wav, _ = sf.read(os.path.join(self.audio_dir, self.ids[i] + ".flac"), start=f * self.hop,
                         stop=(f + self.frames) * self.hop, dtype="float32")
        wav = wav * 10 ** (export_vocoder_audio.HEADROOM_DB / 20)
        return torch.from_numpy(np.array(mel[:, f:f + self.frames], dtype=np.float32)), torch.from_numpy(wav)[None]


def discriminators(h):
    """The discriminators the config trains with, as NVIDIA's train.py builds them (it calls the second one mrd)."""
    from libs.bigvgan import discriminators as d
    if h.get("use_mbd_instead_of_mrd", False):
        mrd = d.MultiBandDiscriminator(h)
    elif h.get("use_cqtd_instead_of_mrd", False):
        mrd = d.MultiScaleSubbandCQTDiscriminator(h)
    else:
        mrd = d.MultiResolutionDiscriminator(h)
    return d.MultiPeriodDiscriminator(h), mrd


def write_run_files(decoder_run, out_dir):
    """OUT_DIR as a decoder run: DECODER_RUN's decoder and stats, and a preprocess.yaml naming the fine-tuned vocoder."""
    for name in ("decoder_best.pt", "stats.json"):
        if not os.path.exists(os.path.join(out_dir, name)):
            shutil.copy(os.path.join(decoder_run, name), os.path.join(out_dir, name))
    pre = decoder_train.read_yaml(os.path.join(decoder_run, "preprocess.yaml"))
    pre["path"]["preprocessed_path"] = os.path.relpath(out_dir, REPO)  # as decoder/train.py writes it
    pre["vocoder"] = {**pre.get("vocoder", {"name": vocoders.DEFAULT}), "checkpoint": "bigvgan_generator.pt"}
    with open(os.path.join(out_dir, "preprocess.yaml"), "w") as f:
        yaml.safe_dump(pre, f, sort_keys=False)
    return pre


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", help="decoder/prepare_data.py's output the decoder was trained on")
    parser.add_argument("audio_dir", help="decoder/export_vocoder_audio.py's output for DATA_DIR")
    parser.add_argument("decoder_run", help="decoder/train.py's run folder (decoder_best.pt, preprocess.yaml, stats.json)")
    parser.add_argument("out_dir", help="where the fine-tuned run goes; re-run to resume")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--segment-frames", type=int, default=64, help="mel frames per training crop (64 = 0.74 s for bigvgan22k)")
    parser.add_argument("--lr", type=float, default=5e-5, help="half of NVIDIA's 1e-4: a fine-tune, at a smaller batch")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--init", default="nvidia", choices=["nvidia", "none"], help="none: random weights (tests)")
    parser.add_argument("--config", help="vocoder config.json instead of the released model's (tests)")
    parser.add_argument("--cache", help="where the decoder's mels are cached (default OUT_DIR/mels)")
    parser.add_argument("--device", default="auto", help="auto (cuda if available, else cpu), cuda, mps or cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    data_dir, audio_dir, decoder_run, out_dir = (os.path.realpath(p) for p in
                                                 (args.data_dir, args.audio_dir, args.decoder_run, args.out_dir))
    cache = os.path.realpath(args.cache) if args.cache else os.path.join(out_dir, "mels")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    os.chdir(REPO)  # WESPER's code reads its configs by relative paths
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True  # fixed crop size

    pre = write_run_files(decoder_run, out_dir)
    name = pre["vocoder"]["name"]
    if name == "hifigan16k":
        sys.exit("only BigVGAN can be fine-tuned: WESPER's HiFi-GAN was released without its discriminators")
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = vocoders.bigvgan_config(name)
    voc = vocoders.bigvgan_spec(name, config)
    from libs.bigvgan.bigvgan import BigVGAN
    from libs.bigvgan.env import AttrDict
    from libs.bigvgan.loss import MultiScaleMelSpectrogramLoss, discriminator_loss, feature_loss, generator_loss
    h = AttrDict(config)
    if not h.get("use_multiscale_melloss", False):
        sys.exit(f"{name} is trained with the single-scale mel loss, which this script doesn't implement")

    # 1. The decoder's mels, cached.
    with open(os.path.join(decoder_run, "stats.json")) as f:
        stats = json.load(f)
    net, model_config = load_decoder(decoder_run, device)
    rows = decoder_train.read_segments(data_dir, model_config["max_seq_len"])
    dataset = decoder_train.Utterances(data_dir, rows, stats, voc)
    started = time.time()
    made = cache_mels(net, dataset, cache, device)
    sample_rows = [r for r in rows if r["split"] == "val"][:SAMPLES]
    sample_mels = [inference_mel(net, dataset[rows.index(r)], device) for r in sample_rows]
    del net
    train_ids = [r["id"] for r in rows if r["split"] == "train" and int(r["frames"]) >= args.segment_frames]
    val_rows = [r for r in rows if r["split"] == "val"]
    print(f"decoder mels: {made} computed in {time.time() - started:.0f} s, {len(rows) - made} cached; "
          f"train: {len(train_ids)} utterances, val: {len(val_rows)}, vocoder: {name}, device: {device}", flush=True)

    # 2. The vocoder, its discriminators and optimizers.
    generator = BigVGAN(h, use_cuda_kernel=False).to(device)
    mpd, mrd = (d.to(device) for d in discriminators(h))
    betas = (h["adam_b1"], h["adam_b2"])
    optim_g = torch.optim.AdamW(generator.parameters(), args.lr, betas=betas)
    optim_d = torch.optim.AdamW(itertools.chain(mrd.parameters(), mpd.parameters()), args.lr, betas=betas)
    latest = os.path.join(out_dir, "latest.pt")
    if os.path.exists(latest):
        state = torch.load(latest, map_location="cpu", weights_only=False)
        for module, key in ((generator, "generator"), (mpd, "mpd"), (mrd, "mrd"), (optim_g, "optim_g"), (optim_d, "optim_d")):
            module.load_state_dict(state[key])
        step, best, history = state["step"], state["best"], state["history"]
        print(f"resuming from step {step}", flush=True)
    else:
        step, best, history = 0, float("inf"), {"train": [], "val": []}
        if args.init == "nvidia":
            generator.load_state_dict(torch.load(vocoders.bigvgan_file(name, "bigvgan_generator.pt"),
                                                 map_location="cpu", weights_only=False)["generator"])
            state = torch.load(vocoders.bigvgan_file(name, "bigvgan_discriminator_optimizer.pt"),
                               map_location="cpu", weights_only=False)
            for module, key in ((mpd, "mpd"), (mrd, "mrd"), (optim_g, "optim_g"), (optim_d, "optim_d")):
                module.load_state_dict(state[key])
            del state
            print(f"starting from NVIDIA's {name}: generator, discriminators and optimizer states", flush=True)
    for optim in (optim_g, optim_d):
        for group in optim.param_groups:
            group["lr"] = args.lr
    mel_loss = MultiScaleMelSpectrogramLoss(sampling_rate=h["sampling_rate"])
    clip = h.get("clip_grad_norm", 1000.0)

    @torch.no_grad()
    def evaluate():
        """Val mel, and the samples at this step."""
        generator.eval()
        errors = []
        for r in val_rows:
            mel = torch.from_numpy(np.load(os.path.join(cache, r["id"] + ".npy")).astype(np.float32))[None].to(device)
            out, _ = vocoders.mel_energy(generator(mel).squeeze().float().cpu().numpy(), voc)
            with np.load(os.path.join(data_dir, "segments", r["id"] + ".npz")) as d:
                target = d["mel"].astype(np.float32)  # the recording's mel (prepare_data.py)
            n = min(out.shape[1], target.shape[1]) - 2  # not the last frames: their windows reach past the cut
            errors.append(float(np.abs(out[:, :n] - target[:, :n]).mean()))
        folder = os.path.join(out_dir, "samples", f"step_{step:06d}")
        os.makedirs(folder, exist_ok=True)
        for r, mel in zip(sample_rows, sample_mels):
            sf.write(os.path.join(folder, r["id"] + ".wav"), generator(mel[None]).squeeze().float().cpu().numpy(),
                     voc.sample_rate, subtype="FLOAT")
        generator.train()
        return float(np.mean(errors))

    reference = os.path.join(out_dir, "samples", "reference")
    if not os.path.exists(reference):
        os.makedirs(reference)
        for r in sample_rows:
            wav, _ = export_vocoder_audio.load(os.path.join(audio_dir, r["id"] + ".flac"))
            sf.write(os.path.join(reference, r["id"] + ".wav"), wav, voc.sample_rate, subtype="FLOAT")
    if step == 0:
        best = evaluate()
        history["val"].append({"step": 0, "mel": best})
        save({"generator": generator.state_dict()}, os.path.join(out_dir, "bigvgan_generator.pt"))
        print(f"  val mel {best:.4f} with the original vocoder", flush=True)
    baseline = history["val"][0]["mel"]

    # 3. Training, as NVIDIA's train.py does it.
    rng = torch.Generator().manual_seed(args.seed + step)
    generator.train(), mpd.train(), mrd.train()
    running, started, start_step = [], time.time(), step
    while step < args.steps:
        loader = torch.utils.data.DataLoader(
            Crops(train_ids, cache, audio_dir, args.segment_frames, voc.hop), batch_size=args.batch_size, shuffle=True,
            drop_last=True, generator=rng, num_workers=args.workers, pin_memory=device == "cuda",
            timeout=300 if args.workers else 0)
        for x, y in loader:
            if step >= args.steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y_g_hat = generator(x)

            optim_d.zero_grad()
            y_df_r, y_df_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, _, _ = discriminator_loss(y_df_r, y_df_g)
            y_ds_r, y_ds_g, _, _ = mrd(y, y_g_hat.detach())
            loss_disc_s, _, _ = discriminator_loss(y_ds_r, y_ds_g)
            loss_disc = loss_disc_s + loss_disc_f
            loss_disc.backward()
            torch.nn.utils.clip_grad_norm_(mpd.parameters(), clip)
            torch.nn.utils.clip_grad_norm_(mrd.parameters(), clip)
            optim_d.step()

            optim_g.zero_grad()
            loss_mel = mel_loss(y, y_g_hat) * h.get("lambda_melloss", 15.0)
            y_df_r, y_df_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_r, y_ds_g, fmap_s_r, fmap_s_g = mrd(y, y_g_hat)
            loss_fm = feature_loss(fmap_f_r, fmap_f_g) + feature_loss(fmap_s_r, fmap_s_g)
            loss_adv = generator_loss(y_df_g)[0] + generator_loss(y_ds_g)[0]
            (loss_adv + loss_fm + loss_mel).backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), clip)
            optim_g.step()
            step += 1
            running.append([loss_mel.item(), loss_fm.item(), loss_adv.item(), loss_disc.item()])

            if step % 100 == 0 or step == args.steps:
                m = np.mean(running, axis=0)
                rate = (step - start_step) / (time.time() - started)
                history["train"].append({"step": step, "mel": m[0], "fm": m[1], "adv": m[2], "disc": m[3]})
                print(f"step {step}/{args.steps}  mel {m[0]:.3f}  fm {m[1]:.3f}  adv {m[2]:.3f}  disc {m[3]:.3f}  "
                      f"{rate:.2f} steps/s  ETA {(args.steps - step) / rate / 60:.0f} min", flush=True)
                running = []
            if step % args.eval_every == 0 or step == args.steps:
                score = evaluate()
                history["val"].append({"step": step, "mel": score})
                improved = score < best
                if improved:
                    best = score
                    save({"generator": generator.state_dict()}, os.path.join(out_dir, "bigvgan_generator.pt"))
                print(f"  val mel {score:.4f} (original vocoder {baseline:.4f})"
                      f"{'  -> saved bigvgan_generator.pt' if improved else ''}", flush=True)
            if step % args.save_every == 0 or step == args.steps:
                save({"generator": generator.state_dict(), "mpd": mpd.state_dict(), "mrd": mrd.state_dict(),
                      "optim_g": optim_g.state_dict(), "optim_d": optim_d.state_dict(), "step": step, "best": best,
                      "history": history, "args": vars(args)}, latest)
                with open(os.path.join(out_dir, "history.json"), "w") as f:
                    json.dump(history, f, indent=1)

    rel = os.path.relpath(out_dir, REPO)
    print(f"done at step {step}. Best val mel {best:.4f}, original vocoder {baseline:.4f}. "
          f"Use it with WESPER: DECODER={rel} ./client_direct_sv.sh")


if __name__ == "__main__":
    main()
