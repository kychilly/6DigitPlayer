"""Makes run_bot.py learn from its own finished plays.

  run_bot.py --policy checkpoints/policy.pt
      |-- starts  learn_from_play.py --watch   (a SEPARATE low-priority process, so its simulator checks
      |                                          can't steal CPU/GIL time from the bot's 1 kHz control loop)
      '-- starts  PolicyReloader               (swaps the improved weights into the running policy)

After each finished map osu! saves a replay -> the learner scores it (300/100/50/miss + reward points),
takes a small reward-weighted step, keeps the update only if the simulator score didn't get worse, and
writes checkpoints/policy.pt atomically -> the reloader loads it, so the NEXT map is played by the
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

    def __init__(self, runner, path: str, log, period: float = 2.0):
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
                time.sleep(0.3)                                   # let the writer finish
                ck = torch.load(self.path, map_location=self.runner.device, weights_only=False)
                with torch.no_grad():
                    self.runner.model.load_state_dict(ck["model"])
                self.runner.model.eval()
                self.last = m
                self.reloads += 1
                self.log(f"policy updated from last map's replay (reload #{self.reloads})")
                print(f"[self-learn] new policy loaded (reload #{self.reloads})", flush=True)
            except Exception as e:  # noqa  (half-written file, arch mismatch...) -> try again next tick
                print(f"[self-learn] reload failed, will retry: {type(e).__name__}: {e}", flush=True)

    def stop(self):
        self._stop_flag = True


class ReplayLearning:
    def __init__(self, a, cfg, runner, shared):
        self.a, self.cfg, self.runner, self.shared = a, cfg, runner, shared
        self.policy_path = os.path.abspath(a.policy)
        self.proc = None
        self.reloader = None

    def start(self):
        a = self.a
        cmd = [sys.executable, "-u", str(HERE / "learn_from_play.py"), "--watch", "--exit-on-stdin-eof",
               "--policy", self.policy_path, "--human-data", os.path.abspath(a.human_data),
               "--lr", str(a.replay_lr), "--eval-episodes", "2", "--threads", "1", "--device", "cpu"]
        if a.songs:
            cmd += ["--songs", a.songs]
        if a.replay_dirs:
            cmd += ["--replay-dirs", *a.replay_dirs]
        if a.player:
            cmd += ["--player", a.player]
        # below-normal priority on Windows so learning never competes with the bot's control loop
        flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
        self.proc = subprocess.Popen(cmd, cwd=str(HERE), stdin=subprocess.PIPE, creationflags=flags)
        self.reloader = PolicyReloader(self.runner, self.policy_path, self.shared.log)
        self.reloader.start()
        self.shared.log("replay learning ON")
        print(f"[self-learn] ON - after every finished map the bot scores its replay and improves "
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
