"""Stage 2 - LEARN BY PLAYING (reinforcement learning in the osu! simulator).

    python train_rl.py --init checkpoints/policy.pt --bc-data data/replays.npz --iters 300
    python train_rl.py --init checkpoints/policy.pt --maps "C:/osu!/Songs" --iters 500   # also play real maps

Reward (config.RewardConfig): 300 = +1.0, 100 = +0.5, 50 = +0.2, miss / off-note = 0 (+ small penalty for
stray clicks), a small bonus for tapping near 0 ms, and dense shaping for aiming at the next note,
following slider balls and spinning. Starting from the replay-cloned policy (--init) is strongly
recommended; from scratch it also works but takes much longer.
"""
import argparse
import os
import time

import numpy as np
import torch

from config import BotConfig
from map_gen import load_maps
from model import pick_device, device_name
from policy import ActorCritic, load_policy, save_policy
from ppo import PPOLearner
from sim_env import OsuSimEnv
from sim_eval import eval_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=None, help="policy checkpoint to start from (e.g. from train_bc.py)")
    ap.add_argument("--out", default="checkpoints/policy_rl.pt")
    ap.add_argument("--bc-data", default=None, help="replay dataset to keep imitating while doing RL")
    ap.add_argument("--bc-coef", type=float, default=0.5)
    ap.add_argument("--maps", default=None, help="folder of real .osu beatmaps to practise on")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--rollout", type=int, default=512, help="steps per env per update")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=5, help="iterations that only train the critic")
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--eval-every", type=int, default=10)
    a = ap.parse_args()

    cfg = BotConfig()
    dev = pick_device()
    print(f"[rl] device {dev} ({device_name(dev)})")
    if a.init and os.path.exists(a.init):
        model = load_policy(a.init, cfg.policy, dev)
        print(f"[rl] starting from {a.init}")
    else:
        model = ActorCritic(cfg.policy).to(dev)
        print("[rl] starting from scratch (consider running train_bc.py first)")
    maps = load_maps(a.maps)
    envs = [OsuSimEnv(cfg, maps, seed=1000 + i, max_seconds=a.max_seconds) for i in range(a.envs)]
    obs = np.stack([e.reset() for e in envs])
    learner = PPOLearner(model, cfg, dev, lr=a.lr, bc_data=a.bc_data, bc_coef=a.bc_coef, ent_coef=a.ent,
                         critic_warmup=a.warmup)
    N, T, D = a.envs, a.rollout, envs[0].obs_dim
    best = -1.0
    ep_stats = []
    for it in range(a.iters):
        t0 = time.time()
        B_o = np.zeros((T, N, D), np.float32); B_a = np.zeros((T, N, 4), np.float32)
        B_lp = np.zeros((T, N), np.float32); B_v = np.zeros((T, N), np.float32)
        B_r = np.zeros((T, N), np.float32); B_d = np.zeros((T, N), np.float32)
        for t in range(T):
            with torch.no_grad():
                act, lp, v = model.act(torch.from_numpy(obs).to(dev), False, 1.0, 1.0)
            act_np = act.cpu().numpy()
            B_o[t], B_a[t], B_lp[t], B_v[t] = obs, act_np, lp.cpu().numpy(), v.cpu().numpy()
            for i, e in enumerate(envs):
                o2, r, done, info = e.step(act_np[i])
                B_r[t, i], B_d[t, i] = r, float(done)
                if done:
                    ep_stats.append((info["acc"], sum(info["counts"][k] for k in ("300", "100", "50", "miss"))))
                    o2 = e.reset()
                obs[i] = o2
        with torch.no_grad():
            last_v = model(torch.from_numpy(obs).to(dev))[3].cpu().numpy()
        st = learner.update(B_o, B_a, B_lp, B_v, B_r, B_d, last_v)
        recent = ep_stats[-16:]
        acc = np.mean([x[0] for x in recent]) if recent else float("nan")
        print(f"iter {it + 1:4d}/{a.iters} | {N * T / (time.time() - t0):5.0f} steps/s | mean r/step {B_r.mean():.4f} | "
              f"recent-episode acc {acc * 100:5.1f}% | pg {st['pg']:+.4f} vf {st['vf']:.3f} kl {st['kl']:+.4f} bc {st['bc']:.2f}",
              flush=True)
        if (it + 1) % a.eval_every == 0 or it + 1 == a.iters:
            r = eval_policy(model, cfg, maps, 6, seed=777)
            print(f"   EVAL acc {r['acc'] * 100:.1f}%  {r['counts']}  tap offset {r['tap_offset_ms_mean']:+.1f} "
                  f"+/- {r['tap_offset_ms_std']:.1f} ms  return {r['return']:.1f}", flush=True)
            save_policy(a.out.replace(".pt", "_latest.pt"), model, {"stage": "rl", "iter": it + 1})
            if r["acc"] > best:
                best = r["acc"]
                save_policy(a.out, model, {"stage": "rl", "iter": it + 1, "eval_acc": best})
                print(f"   saved {a.out} (best eval acc {best * 100:.1f}%)")


if __name__ == "__main__":
    main()
