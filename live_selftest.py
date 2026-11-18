"""Tests the LIVE controller (PolicyController) without osu!: a fake 'screen tracker' shows a simulated
map, the controller runs at 1 kHz on a virtual clock, and a fake keyboard/mouse feeds the real osu!
judge so we can compare what the bot THINKS happened (its self-judged reward) with the truth.

    python live_selftest.py --policy checkpoints/policy.pt [--learn]
"""
from __future__ import annotations
import argparse
import math

import numpy as np
import torch

from config import BotConfig, ar_to_preempt
from judge import Judge
from map_gen import random_beatmap
from live_learning import OnlineLearner
from playfield import PlayfieldMap
from policy import PolicyRunner
from policy_controller import PolicyController
from state import SharedState
from tracker import TrackedObject, BallState
from osu_formats import SLIDER, SPINNER


class FakeWorld:
    """Acts as tracker + mouse/keyboard backend for one simulated map on a virtual clock."""

    def __init__(self, cfg, bm):
        self.cfg, self.bm = cfg, bm
        self.pfm = PlayfieldMap(cfg.capture, cfg.playfield)
        self.preempt = ar_to_preempt(bm.ar)
        self.now = 0.0
        self.judge = Judge(bm.objects, bm.od, bm.radius)       # ground truth for what the bot actually did
        self.gone: set = set()
        self.pos_screen = np.array([960.0, 540.0])
        self.keys = {k: False for k in cfg.game.keys}
        self.rng = np.random.default_rng(0)
        self.bias = {}

    # ---- tracker interface
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
            out.append(TrackedObject(o.index, cls, fx, fy, self.bm.radius * self.pfm.s, o.time + b[2],
                                     self.now - 0.3, self.now, hits=5, verified=True))
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
                return BallState(fx, fy, self.now, float(v[0]), float(v[1]))
        return None

    def mark_clicked(self, oid, now):
        self.gone.add(oid)

    def mark_active(self, oid): pass

    # ---- backend interface
    def get_pos(self): return tuple(self.pos_screen)

    def move(self, x, y): self.pos_screen = np.array([x, y])

    def key_down(self, k):
        was = any(self.keys.values())
        self.keys[k] = True
        self.judge.press(self.now, self.pfm.screen_to_pf(*self.pos_screen))

    def key_up(self, k): self.keys[k] = False


def run_map(cfg, runner, learner, seed, seconds=25.0):
    rng = np.random.default_rng(seed)
    bm = random_beatmap(rng, seconds)
    cfg.game.ar, cfg.game.od = bm.ar, bm.od
    world = FakeWorld(cfg, bm)
    shared = SharedState(); shared.armed = True
    world.now = bm.objects[0].time - world.preempt - 0.3
    ctl = PolicyController(cfg, world, None, world, shared, runner, learner)
    end = max(o.end_time for o in bm.objects) + 5.0
    last = world.now
    while world.now < end and not shared.quit:
        world.now += 0.001
        world.judge.advance(world.now)
        ctl.step(world.now, 0.001)
        if world.now - last > 5:
            last = world.now
    ctl._disengage("test end")
    return world.judge, ctl.judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="checkpoints/policy.pt")
    ap.add_argument("--learn", action="store_true")
    ap.add_argument("--maps", type=int, default=2)
    a = ap.parse_args()
    cfg = BotConfig()
    cfg.gate.idle_release = 0.5
    runner = PolicyRunner(cfg.policy, torch.device("cpu"), a.policy)
    learner = None
    if a.learn:
        cfg.policy.key_temp = cfg.policy.move_noise_scale = 1.0
        learner = OnlineLearner(runner, cfg, save_path="/tmp/live_test.pt", min_steps=300, log=print)
    for s in range(a.maps):
        truth, believed = run_map(cfg, runner, learner, 100 + s)
        print(f"map {s}: TRUE judge {truth.counts} acc {truth.accuracy() * 100:.1f}% | "
              f"bot's own live judge {believed.counts} acc {believed.accuracy() * 100:.1f}%")


if __name__ == "__main__":
    main()
