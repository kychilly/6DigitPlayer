"""Makes run_bot.py learn after every map (true reinforcement learning).

  run_bot.py --policy checkpoints/policy.pt
      |-- the bot records each map it plays             -> data/rl_episodes/*.npz   (live_learning.EpisodeRecorder)
      |-- starts  rl_worker.py --watch                 (a SEPARATE low-priority process, so PPO can't steal CPU/GIL
      |                                                  time from the bot's 1 kHz control loop)
      '-- starts  PolicyReloader                       (swaps the improved weights into the running policy)

After each map: the worker finds the replay osu! saved, scores every step with osu!'s real judgement, runs a PPO
update, and writes checkpoints/policy.pt atomically -> the reloader loads it, so the NEXT map is played by the
improved policy. No restart needed.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


class PolicyReloader(threading.Thread):
    """Watches the policy file; when it changes, loads the new weights into the live model IN PLACE
    (so it works no matter who holds a reference to the model)."""

    def __init__(self, runner, path: str, log, period: float = 1.0):
        super().__init__(daemon=True)
        self.runner, self.path, self.log, self.period = runner, path, log, period
        self.last = self._mtime()
        self._stop_flag = False
        self.reloads = 0

    def _mtime(self):
        try:
            return os.path.getmtime(self.path)
        except OSError:
            return None

    def run(self):
        while not self._stop_flag:
            time.sleep(self.period)
            m = self._mtime()
            if m is None or m == self.last:
                continue
            try:
                time.sleep(0.2)                                   # let the writer finish
                ck = torch.load(self.path, map_location=self.runner.device, weights_only=False)
                with self.runner.lock, torch.no_grad():         # never swap weights mid-decision
                    self.runner.model.load_state_dict(ck["model"])
                self.runner.model.eval()
                self.last = m
                self.reloads += 1
                n = ck.get("extra", {}).get("updates", "?")
                self.log(f"policy improved from last map (update #{n}) - loaded")
                print(f"[self-learn] new policy loaded (update #{n})", flush=True)
            except Exception as e:  # noqa  (half-written file, arch mismatch...) -> try again next tick
                print(f"[self-learn] reload failed, will retry: {type(e).__name__}: {e}", flush=True)

    def stop(self):
        self._stop_flag = True


class ReplayLearning:
    """Starts rl_worker.py next to the bot and hot-reloads what it learns."""

    def __init__(self, a, cfg, runner, shared):
        self.a, self.cfg, self.runner, self.shared = a, cfg, runner, shared
        self.policy_path = os.path.abspath(a.policy)
        self.proc = None
        self.reloader = None

    def start(self):
        a = self.a
        cmd = [sys.executable, "-u", str(HERE / "rl_worker.py"), "--watch", "--exit-on-stdin-eof",
               "--policy", self.policy_path, "--max-replays", str(a.max_replays),
               "--lr", str(a.rl_lr), "--threads", "1", "--device", "cpu"]
        if a.human_data:
            cmd += ["--human-data", os.path.abspath(a.human_data), "--bc-coef", str(a.bc_coef)]
        if a.songs:
            cmd += ["--songs", a.songs]
        if a.replay_dirs:
            cmd += ["--replay-dirs", *a.replay_dirs]
        if a.no_live_fallback:
            cmd += ["--no-live-fallback"]
        # below-normal priority on Windows so learning never competes with the bot's control loop
        flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
        self.proc = subprocess.Popen(cmd, cwd=str(HERE), stdin=subprocess.PIPE, creationflags=flags)
        self.reloader = PolicyReloader(self.runner, self.policy_path, self.shared.log)
        self.reloader.start()
        self.shared.log("RL after every map: ON")
        print(f"[self-learn] ON - after every map the bot scores its replay with osu!'s rules and runs PPO on "
              f"{self.policy_path}", flush=True)

    def stop(self):
        if self.reloader:
            self.reloader.stop()
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.terminate()
            except Exception:  # noqa
                pass
