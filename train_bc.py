"""Stage 1 - LEARN FROM REPLAYS (imitation / behaviour cloning).

    python replay_data.py --replays <osu!/Data/r> --songs <osu!/Songs> --out data/replays.npz
    python train_bc.py --data data/replays.npz --epochs 30

The policy learns to move the cursor and press keys the way the human did, given only what was on
screen. Mirror flips + perception noise are applied to the inputs so it does not just copy its own
last velocity and so it tolerates real detector noise. Prints sim accuracy after every few epochs.
"""
import argparse
import time

import numpy as np
import torch

from config import BotConfig
from model import pick_device, device_name
from policy import ActorCritic, bc_loss, augment, save_policy
from sim_eval import eval_policy
from map_gen import load_maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/replays.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="checkpoints/policy.pt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--maps", default=None, help="folder of .osu files to also evaluate on in the simulator")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-episodes", type=int, default=4)
    ap.add_argument("--device", default="cpu", help="cpu (recommended), xpu or cuda")
    a = ap.parse_args()

    cfg = BotConfig()
    dev = torch.device(a.device)
    print(f"[bc] device {dev} ({device_name(dev)})")
    d = np.load(a.data)
    assert int(d["n_objs"]) == cfg.policy.n_objs and int(d["hz"]) == cfg.policy.hz, "dataset/config mismatch - rebuild it"
    obs = torch.from_numpy(d["obs"]).to(dev)
    act = torch.from_numpy(d["act"]).to(dev)
    N = len(obs)
    n_val = max(1000, N // 20)
    tr_o, tr_a, va_o, va_a = obs[:-n_val], act[:-n_val], obs[-n_val:], act[-n_val:]   # last 5% = held-out
    print(f"[bc] {N} samples ({len(tr_o)} train / {n_val} val)")

    model = ActorCritic(cfg.policy).to(dev)
    if a.resume:
        model.load_state_dict(torch.load(a.resume, map_location=dev, weights_only=False)["model"])
    opt = torch.optim.AdamW(list(model.actor.parameters()) + list(model.mu.parameters()) +
                            list(model.logstd.parameters()) + list(model.keys.parameters()), lr=a.lr, weight_decay=1e-4)
    steps_per = max(1, len(tr_o) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps_per)
    maps = load_maps(a.maps)
    best = 1e9
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr_o), device=dev)
        t0, run = time.time(), {"nll": 0, "ce": 0, "mse": 0, "key_acc": 0}
        for s in range(steps_per):
            i = perm[s * a.batch:(s + 1) * a.batch]
            o, y = tr_o[i], tr_a[i]
            if not a.no_aug:
                o, y = augment(o, y, cfg.policy)
            loss, parts = bc_loss(model, o, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            for k in run:
                run[k] += parts[k] / steps_per
        model.eval()
        with torch.no_grad():
            vl, vp = bc_loss(model, va_o, va_a)
        print(f"epoch {ep + 1:3d}/{a.epochs} train nll {run['nll']:.3f} ce {run['ce']:.3f} | "
              f"val nll {vp['nll']:.3f} ce {vp['ce']:.3f} key-acc {vp['key_acc']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if float(vl) < best:
            best = float(vl)
            save_policy(a.out, model, {"stage": "bc", "epoch": ep + 1})
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            r = eval_policy(model, cfg, maps, a.eval_episodes)
            print(f"   sim play: acc {r['acc'] * 100:.1f}%  {r['counts']}  tap offset {r['tap_offset_ms_mean']:+.1f}"
                  f" +/- {r['tap_offset_ms_std']:.1f} ms", flush=True)
    print(f"[bc] best checkpoint: {a.out}")


if __name__ == "__main__":
    main()
