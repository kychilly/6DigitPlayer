"""Learning from the bot's own plays.

While a map is played, PolicyController fills an ExperienceBuffer with exactly what the policy saw and
did (observation, action, log-prob, value, the bot's own guess of the reward, the time of each step and
which keys went down). When the map ends, EpisodeRecorder writes that buffer to data/rl_episodes/.

rl_worker.py (started automatically by run_bot.py) then:
  1. finds the replay osu! saved for that play,
  2. re-scores every step with osu!'s REAL judgement (300/100/50/miss, stray clicks, slider follow,
     spinner) from the replay + the beatmap,
  3. runs a PPO update on that episode (true reinforcement learning), and
  4. writes the improved policy, which the running bot hot-reloads before the next map.

OnlineLearner (in-process PPO on the bot's self-judged reward) is kept for live_selftest / experiments.
"""
from __future__ import annotations
import os
import threading
import time
from pathlib import Path

import numpy as np

EPISODE_DIR = Path("data/rl_episodes")


class ExperienceBuffer:
    def __init__(self):
        self.o, self.a, self.lp, self.v, self.r = [], [], [], [], []
        self.t, self.press = [], []          # step time (perf_counter s) and which keys went DOWN this step

    def add(self, obs, act, logp, val, rew, t: float = 0.0, press=(False, False)):
        self.o.append(obs); self.a.append(act); self.lp.append(logp); self.v.append(val); self.r.append(rew)
        self.t.append(t); self.press.append((bool(press[0]), bool(press[1])))

    def __len__(self):
        return len(self.r)

    def arrays(self):
        T = len(self.r)
        done = np.zeros((T, 1), np.float32); done[-1] = 1.0
        return (np.asarray(self.o, np.float32)[:, None], np.asarray(self.a, np.float32)[:, None],
                np.asarray(self.lp, np.float32)[:, None], np.asarray(self.v, np.float32)[:, None],
                np.asarray(self.r, np.float32)[:, None], done)


class EpisodeRecorder:
    """Drop-in for OnlineLearner: instead of learning in the bot process, it saves the episode for rl_worker.py
    (learning in a separate process keeps the 1 kHz control loop smooth)."""

    def __init__(self, cfg, out_dir: str | Path = EPISODE_DIR, min_steps: int = 600, log=print):
        self.cfg, self.out_dir, self.min_steps, self.log = cfg, Path(out_dir), min_steps, log

    def finish_episode(self, buf: ExperienceBuffer, acc: float, counts: dict, meta: dict | None = None,
                       sync: bool = False):
        if len(buf) < self.min_steps:
            self.log(f"learn: episode too short ({len(buf)} steps), not saved")
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        name = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        path = self.out_dir / f"{name}.npz"
        meta = dict(meta or {})
        data = dict(obs=np.asarray(buf.o, np.float32), act=np.asarray(buf.a, np.float32),
                    logp=np.asarray(buf.lp, np.float32), val=np.asarray(buf.v, np.float32),
                    rew_live=np.asarray(buf.r, np.float32), t=np.asarray(buf.t, np.float64),
                    press=np.asarray(buf.press, bool),
                    wall_end=np.float64(time.time()), live_acc=np.float32(acc),
                    live_counts=np.array([counts.get(k, 0) for k in ("300", "100", "50", "miss", "stray")]),
                    ar=np.float32(meta.get("ar", self.cfg.game.ar)), od=np.float32(meta.get("od", self.cfg.game.od)),
                    map_name=np.array(str(meta.get("map", ""))),
                    n_objs=np.int32(self.cfg.policy.n_objs), hz=np.int32(self.cfg.policy.hz))

        def write():                       # off the control thread; write-then-rename so the worker never reads half a file
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **data)
            os.replace(tmp, path)
        if sync:
            write()                        # kill switch / shutdown: the process exits right after this
        else:
            threading.Thread(target=write, daemon=True).start()
        self.log(f"learn: saved episode ({len(buf)} steps) -> worker will score it with the replay")
        return path


class OnlineLearner:
    """In-process PPO on the bot's OWN (self-judged) reward. Kept for live_selftest / experiments;
    run_bot.py uses EpisodeRecorder + rl_worker.py, which learn from osu!'s real judgement instead."""

    def __init__(self, runner, cfg, lr=3e-5, bc_data=None, bc_coef=0.5, save_path="checkpoints/policy_live.pt",
                 min_steps=600, log=print):
        from ppo import PPOLearner
        self.runner, self.cfg, self.save_path, self.min_steps, self.log = runner, cfg, save_path, min_steps, log
        self.ppo = PPOLearner(runner.model, cfg, runner.device, lr=lr, bc_data=bc_data, bc_coef=bc_coef,
                              critic_warmup=2)
        self.best_acc = -1.0
        self.episodes = 0

    def finish_episode(self, buf: ExperienceBuffer, acc: float, counts: dict, meta: dict | None = None, **_):
        from policy import save_policy
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
