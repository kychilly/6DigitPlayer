"""Play the simulator with a policy and report judgement stats (no learning)."""
from __future__ import annotations
import numpy as np
import torch

from config import BotConfig
from sim_env import OsuSimEnv


@torch.no_grad()
def eval_policy(model, cfg: BotConfig, maps=None, episodes: int = 5, seed: int = 123,
                deterministic: bool = False, max_seconds: float = 30.0, device=None):
    device = device or next(model.parameters()).device
    env = OsuSimEnv(cfg, maps, seed=seed, max_seconds=max_seconds)
    was_training = model.training
    model.eval()
    tot = {"300": 0, "100": 0, "50": 0, "miss": 0, "stray": 0}
    ret, offs = 0.0, []
    for _ in range(episodes):
        o = env.reset()
        done = False
        while not done:
            a, _, _ = model.act(torch.from_numpy(o).unsqueeze(0).to(device), deterministic,
                                cfg.policy.key_temp, cfg.policy.move_noise_scale)
            o, r, done, info = env.step(a[0].cpu().numpy())
            ret += r
        for k in tot:
            tot[k] += info["counts"][k]
        offs += [v * 1000 for v in env.judge.dts.values() if abs(v) < 0.2]
    model.train(was_training)
    n = tot["300"] + tot["100"] + tot["50"] + tot["miss"]
    acc = (300 * tot["300"] + 100 * tot["100"] + 50 * tot["50"]) / (300.0 * max(n, 1))
    return {"acc": acc, "counts": tot, "return": ret / episodes,
            "tap_offset_ms_mean": float(np.mean(offs)) if offs else 0.0,
            "tap_offset_ms_std": float(np.std(offs)) if offs else 0.0}
