"""Tests the LIVE controller (PolicyController) + the after-every-map RL loop without osu!.

A fake 'screen tracker' shows a simulated map, the controller runs at 1 kHz on a virtual clock, and a fake
keyboard/mouse feeds the real osu! judge. The fake world also writes a real .osu file and a real .osr replay of
what the bot did (on a DIFFERENT clock than the bot's, like real osu!), so rl_worker.py has to find the replay,
line it up with the bot's own recording, score it and run PPO - exactly what happens after a real map.

    python live_selftest.py --policy checkpoints/policy.pt               # just play 2 maps, compare judges
    python live_selftest.py --policy checkpoints/policy.pt --rl --maps 30 # play -> learn -> play ... and watch acc
"""
from __future__ import annotations
import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import BotConfig, ar_to_preempt
from judge import Judge
from map_gen import random_beatmap
from live_learning import EpisodeRecorder
from osu_formats import parse_osu, SLIDER, SPINNER
from playfield import PlayfieldMap
from policy import PolicyRunner
from policy_controller import PolicyController
from state import SharedState
from tracker import TrackedObject, BallState


class FakeWorld:
    """Acts as tracker + mouse/keyboard backend for one simulated map.
    self.now is MAP time (what the replay uses); the bot sees perf-counter time = map time + clock_offset."""

    def __init__(self, cfg, bm, clock_offset=0.0, seed=0):
        self.cfg, self.bm, self.off = cfg, bm, clock_offset
        self.pfm = PlayfieldMap(cfg.capture, cfg.playfield)
        self.preempt = ar_to_preempt(bm.ar)
        self.now = 0.0
        self.judge = Judge(bm.objects, bm.od, bm.radius)       # ground truth for what the bot actually did
        self.gone: set = set()
        self.pos_screen = np.array([960.0, 540.0])
        self.keys = {k: False for k in cfg.game.keys}
        self.rng = np.random.default_rng(seed)
        self.bias = {}
        self.frames = []                                      # replay frames (map time, pf x, pf y, key bits)
        self._last_frame = -1.0

    # ---- tracker interface (times in the BOT's clock)
    def _objs(self):
        out = []
        for o in self.bm.objects:
            if o.index in self.gone or o.time - self.preempt > self.now or o.end_time + 0.3 < self.now:
                continue
            if o.kind == SPINNER:
                if self.now > o.end_time:
                    continue
            elif o.index in self.judge.resolved:
                continue
            b = self.bias.setdefault(o.index, self.rng.normal(0, [1.5, 1.5, 0.006]))
            fx, fy = self.pfm.pf_to_frame(o.x + b[0], o.y + b[1])
            cls = 2 if o.kind == SPINNER else (1 if o.kind == SLIDER else 0)
            out.append(TrackedObject(o.index, cls, fx, fy, self.bm.radius * self.pfm.s, o.time + b[2] + self.off,
                                     self.now - 0.3 + self.off, self.now + self.off, hits=5, verified=True))
        return out

    def pending(self):
        return sorted(self._objs(), key=lambda o: o.hit_time)

    def has_targets(self): return bool(self._objs())
    def has_verified(self): return bool(self._objs())

    def ball(self):
        for o in self.bm.objects:
            if o.kind == SLIDER and o.time <= self.now <= o.end_time:
                b, b2 = o.ball_pos(self.now), o.ball_pos(self.now + 0.01)
                fx, fy = self.pfm.pf_to_frame(*b)
                v = (b2 - b) / 0.01 * self.pfm.s
                return BallState(fx, fy, self.now + self.off, float(v[0]), float(v[1]))
        return None

    def mark_clicked(self, oid, now): self.gone.add(oid)
    def mark_active(self, oid): pass
    def set_preempt(self, p): pass

    # ---- backend interface
    def get_pos(self): return tuple(self.pos_screen)

    def move(self, x, y): self.pos_screen = np.array([x, y])

    def _bits(self):
        k = self.cfg.game.keys
        return (5 if self.keys[k[0]] else 0) | (10 if self.keys[k[1]] else 0)    # K1 = 4|1, K2 = 8|2 (as osu! writes)

    def record(self, force=False):
        if force or self.now - self._last_frame >= 0.008 - 1e-9:
            x, y = self.pfm.screen_to_pf(*self.pos_screen)
            self.frames.append((self.now, x, y, self._bits()))
            self._last_frame = self.now

    def key_down(self, k):
        self.keys[k] = True
        self.judge.press(self.now, self.pfm.screen_to_pf(*self.pos_screen))
        self.record(force=True)

    def key_up(self, k):
        self.keys[k] = False
        self.record(force=True)

    def osr_frames(self):
        out, last = [], -0.001
        for t, x, y, z in self.frames:
            w = int(round((t - last) * 1000))
            if w <= 0:                                    # same millisecond: keep the newest state, never lose a press
                if out:
                    pw, _, _, pz = out[-1]
                    out[-1] = (pw, x, y, z | (pz if (pz & ~z) == 0 else z))
                continue
            last += w / 1000.0
            out.append((w, x, y, z))
        return out


def run_map(cfg, runner, learner, bm, seed, clock_offset=0.0):
    cfg.game.ar, cfg.game.od = bm.ar, bm.od
    world = FakeWorld(cfg, bm, clock_offset, seed)
    shared = SharedState(); shared.armed = True
    world.now = bm.objects[0].time - world.preempt - 0.3
    ctl = PolicyController(cfg, world, None, world, shared, runner, learner)
    ctl.set_difficulty(bm.ar, bm.od, bm.title)
    end = max(o.end_time for o in bm.objects) + 5.0
    while world.now < end and not shared.quit:
        world.now += 0.001
        world.judge.advance(world.now)
        ctl.step(world.now + clock_offset, 0.001)
        world.record()
    ctl._disengage("test end")
    counts, acc = dict(ctl.judge.counts), ctl.judge.accuracy()      # the bot's own view, before the flush resets it
    ctl.flush_episode()
    return world, SimpleNamespace(counts=counts, accuracy=lambda: acc)


def make_map(rng, seconds, songs_dir: Path, i: int):
    """random map -> real .osu file -> parsed back (so the worker can find it by md5 like a real map)."""
    from selftest import write_osu
    bm0 = random_beatmap(rng, seconds)
    p = songs_dir / f"generated_{i:03d}.osu"
    write_osu(bm0, p)
    txt = p.read_text().replace("[Difficulty]", f"[Metadata]\nTitle:generated {i}\n\n[Difficulty]")
    p.write_text(txt)
    return parse_osu(p), hashlib.md5(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="checkpoints/policy.pt")
    ap.add_argument("--rl", action="store_true", help="full loop: play -> replay -> rl_worker PPO -> reload -> play")
    ap.add_argument("--maps", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--clock-offset", type=float, default=12345.678, help="bot clock minus replay clock (s)")
    ap.add_argument("--same-map", action="store_true", help="replay the same map every time (easier to see learning)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--keep", action="store_true", help="keep the temp folder")
    a = ap.parse_args()
    cfg = BotConfig()
    cfg.gate.idle_release = 0.5
    policy_src = os.path.abspath(a.policy)
    human = os.path.abspath("data/replays.npz")

    if not a.rl:
        runner = PolicyRunner(cfg.policy, torch.device("cpu"), policy_src)
        rng = np.random.default_rng(100)
        for s in range(a.maps):
            bm = random_beatmap(rng, a.seconds)
            world, believed = run_map(cfg, runner, None, bm, s)
            print(f"map {s}: TRUE judge {world.judge.counts} acc {world.judge.accuracy() * 100:.1f}% | "
                  f"bot's own live judge {believed.counts} acc {believed.accuracy() * 100:.1f}%")
        return

    # ---------------------------------------------------------------- full RL loop in a sandbox folder
    import rl_worker
    from selftest import write_osr
    work = Path(tempfile.mkdtemp(prefix="rl_selftest_"))
    (work / "checkpoints").mkdir(); songs = work / "songs"; songs.mkdir(); reps = work / "replays"; reps.mkdir()
    shutil.copy(policy_src, work / "checkpoints" / "policy.pt")
    if os.path.exists(human):
        (work / "data").mkdir(); shutil.copy(human, work / "data" / "replays.npz")
    os.chdir(work)
    print(f"[selftest] sandbox: {work}")
    wa = argparse.Namespace(policy="checkpoints/policy.pt", songs=str(songs), replay_dirs=[str(reps)],
                            human_data="data/replays.npz", bc_coef=0.5, lr=a.lr, epochs=6, minibatch=1024,
                            target_kl=0.02, ent=0.0, warmup=2, wait_replay=0.0, live_fallback=True,
                            keep_history=30, keep_episodes=50, device="cpu", extra_songs=[], max_replays=300,
                            study_steps=1000, awr_steps=200, awr_lr=3e-4, awr_beta=1.0)
    worker = rl_worker.Worker(wa)
    runner = PolicyRunner(cfg.policy, torch.device("cpu"), "checkpoints/policy.pt")
    rec = EpisodeRecorder(cfg, min_steps=300, log=lambda m: None)
    rng = np.random.default_rng(7)
    fixed = make_map(rng, a.seconds, songs, 0) if a.same_map else None
    accs = []
    for i in range(a.maps):
        bm, md5 = fixed if fixed else make_map(rng, a.seconds, songs, i)
        world, believed = run_map(cfg, runner, rec, bm, 1000 + i, a.clock_offset)
        write_osr(reps / f"play_{i:03d}.osr", world.osr_frames(), md5=md5)
        old = time.time() - 5
        os.utime(reps / f"play_{i:03d}.osr", (old, old))       # "fully written"
        for _ in range(100):                                    # recorder writes in a thread
            if worker.pending():
                break
            time.sleep(0.05)
        ep = worker.pending()[0]
        e = dict(np.load(ep))
        res = rl_worker.align(e["t"][e["press"].any(1)], rl_worker.replay_presses(
            rl_worker.parse_osr(str(reps / f"play_{i:03d}.osr")))[:, 0])
        truth = world.judge
        print(f"\n### map {i}: TRUE {truth.counts} acc {truth.accuracy() * 100:.1f}% | "
              f"clock offset found {-res[0]:.4f}s (true {a.clock_offset}s), {res[1]} presses matched")
        worker.run_once(wait_for_replay=0.0)
        with runner.lock, torch.no_grad():                      # what PolicyReloader does in the real bot
            runner.model.load_state_dict(torch.load("checkpoints/policy.pt", weights_only=False)["model"])
        accs.append(truth.accuracy())
    print("\nTRUE accuracy per map:", " ".join(f"{x * 100:.0f}" for x in accs))
    if len(accs) >= 6:
        k = len(accs) // 3
        print(f"first {k}: {np.mean(accs[:k]) * 100:.1f}%  last {k}: {np.mean(accs[-k:]) * 100:.1f}%")
    rl_worker.stats()
    if not a.keep:
        os.chdir(HERE)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
