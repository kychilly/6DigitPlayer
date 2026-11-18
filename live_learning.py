"""Learning while playing the real game: experience buffer + PPO update after each map."""
from __future__ import annotations
import os
import time

import numpy as np

from policy import save_policy
from ppo import PPOLearner


class ExperienceBuffer:
    def __init__(self):
        self.o, self.a, self.lp, self.v, self.r = [], [], [], [], []

    def add(self, obs, act, logp, val, rew):
        self.o.append(obs); self.a.append(act); self.lp.append(logp); self.v.append(val); self.r.append(rew)

    def __len__(self):
        return len(self.r)

    def arrays(self):
        T = len(self.r)
        done = np.zeros((T, 1), np.float32); done[-1] = 1.0
        return (np.asarray(self.o, np.float32)[:, None], np.asarray(self.a, np.float32)[:, None],
                np.asarray(self.lp, np.float32)[:, None], np.asarray(self.v, np.float32)[:, None],
                np.asarray(self.r, np.float32)[:, None], done)


class OnlineLearner:
    """Called when a map ends: PPO-updates the same network the bot is playing with."""

    def __init__(self, runner, cfg, lr=3e-5, bc_data=None, bc_coef=0.5, save_path="checkpoints/policy_live.pt",
                 min_steps=600, log=print):
        self.runner, self.cfg, self.save_path, self.min_steps, self.log = runner, cfg, save_path, min_steps, log
        self.ppo = PPOLearner(runner.model, cfg, runner.device, lr=lr, bc_data=bc_data, bc_coef=bc_coef,
                              critic_warmup=2)
        self.best_acc = -1.0
        self.episodes = 0

    def finish_episode(self, buf: ExperienceBuffer, acc: float, counts: dict):
        if len(buf) < self.min_steps:
            self.log(f"learn: episode too short ({len(buf)} steps), skipped")
            return
        t0 = time.time()
        o, a, lp, v, r, d = buf.arrays()
        st = self.ppo.update(o, a, lp, v, r, d, np.zeros(1, np.float32))
        self.episodes += 1
        save_policy(self.save_path.replace(".pt", "_latest.pt"), self.runner.model,
                    {"stage": "live", "episodes": self.episodes})
        if acc >= self.best_acc:
            self.best_acc = acc
            save_policy(self.save_path, self.runner.model, {"stage": "live", "episodes": self.episodes, "acc": acc})
        self.log(f"learn: map #{self.episodes} acc {acc * 100:.1f}% {counts} | reward {float(r.sum()):.1f} | "
                 f"update {time.time() - t0:.1f}s pg {st['pg']:+.4f} vf {st['vf']:.3f}")
