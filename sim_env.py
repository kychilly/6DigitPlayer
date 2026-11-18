"""A headless osu!std simulator. The agent plays it with the same observation / action interface
as the live bot:  action = [vx, vy, key0_down, key1_down]  (vx,vy in units of max_speed).

Rewards come from osu!'s own judgement rules (judge.py): 300 = full reward, 100/50 = less,
misses and stray clicks = nothing / a small penalty, plus dense shaping for aiming and sliders.
"""
from __future__ import annotations
import math
from typing import Optional

import numpy as np

from config import BotConfig, ar_to_preempt
from judge import Judge, RewardShaper, spinner_required_revs
from map_gen import random_beatmap, segment
from obs import VisObj, VisBall, CIRCLE, SLIDER_HEAD, SPINNER as SPIN_KIND, encode, obs_dim, sort_visible
from osu_formats import Beatmap, SLIDER, SPINNER


class OsuSimEnv:
    def __init__(self, cfg: BotConfig, maps: Optional[list] = None, seed: Optional[int] = None,
                 max_seconds: float = 40.0, perception_noise: bool = True, p_real: float = 0.5):
        self.cfg, self.pc, self.rc = cfg, cfg.policy, cfg.reward
        self.maps, self.max_seconds, self.noise, self.p_real = maps or [], max_seconds, perception_noise, p_real
        self.rng = np.random.default_rng(seed)
        self.dt = 1.0 / self.pc.hz
        self.obs_dim = obs_dim(self.pc.n_objs)
        self.bm: Optional[Beatmap] = None

    # ------------------------------------------------------------------ episode
    def reset(self, beatmap: Optional[Beatmap] = None) -> np.ndarray:
        rng = self.rng
        if beatmap is None:
            if self.maps and rng.random() < self.p_real:
                beatmap = segment(self.maps[int(rng.integers(len(self.maps)))], self.max_seconds, rng)
            else:
                beatmap = random_beatmap(rng, self.max_seconds)
        self.bm = bm = beatmap
        self.preempt = ar_to_preempt(bm.ar)
        self.od, self.radius = bm.od, bm.radius
        self.judge = Judge(bm.objects, bm.od, self.radius)
        self.shaper = RewardShaper(self.rc, bm.od)
        self.objs = bm.objects
        self.end_t = max(o.end_time for o in self.objs) + 0.6
        self.t = self.objs[0].time - self.preempt - 0.25
        self.ptr = 0                         # next object to become visible
        self.live: list = []                 # HitObjects currently visible
        self.pos = np.array([256.0, 192.0]) + rng.normal(0, 40, 2)
        self.vel = np.zeros(2)
        self.keys = [False, False]
        self.key_age = [1.0, 1.0]
        self.bias: dict[int, np.ndarray] = {}
        self.sliders = [o for o in self.objs if o.kind == SLIDER]
        self.steps = 0
        self.shaper.reset()
        self._last_obs_objs: list = []
        return self._observe()

    # ------------------------------------------------------------------ stepping
    def step(self, action):
        dt = self.dt
        a = np.asarray(action, float)
        v_des = a[:2] * self.pc.max_speed
        dv = v_des - self.vel
        lim = self.pc.max_accel * dt
        n = float(np.linalg.norm(dv))
        if n > lim:
            dv *= lim / n
        self.vel = self.vel + dv
        self.pos = np.clip(self.pos + self.vel * dt, [-30.0, -20.0], [542.0, 404.0])
        self.t += dt
        self.steps += 1

        new_keys = [bool(a[2] > 0.5), bool(a[3] > 0.5)]
        rising = [n_ and not o_ for n_, o_ in zip(new_keys, self.keys)]
        for i in range(2):
            self.key_age[i] = 0.0 if new_keys[i] != self.keys[i] else self.key_age[i] + dt
        self.keys = new_keys

        events = []
        if any(rising):
            events += self.judge.press(self.t, self.pos)
        else:
            events += self.judge.advance(self.t)
        follow_ok, revs, spin = self.judge.track(self.t, dt, self.pos, any(self.keys))

        vis_true = self._visible_true()
        need = spinner_required_revs(spin.end_time - spin.time, self.od) if spin is not None else 0.0
        r = self.shaper.step(events, self.pos, vis_true, self.keys, follow_ok, revs,
                             spin.index if spin is not None else None, need)
        done = self.t >= self.end_t
        info = {}
        if done:
            info = {"counts": dict(self.judge.counts), "acc": self.judge.accuracy(), "n_objs": len(self.objs)}
        return self._observe(vis_true), float(r), done, info

    # ------------------------------------------------------------------ views
    def _visible_true(self) -> list:
        t = self.t
        while self.ptr < len(self.objs) and self.objs[self.ptr].time - self.preempt <= t:
            self.live.append(self.objs[self.ptr])
            self.ptr += 1
        keep, vis = [], []
        for o in self.live:
            if o.kind == SPINNER:
                if t > o.end_time:
                    continue
                kind = SPIN_KIND
            else:
                if o.index in self.judge.resolved:
                    continue
                kind = SLIDER_HEAD if o.kind == SLIDER else CIRCLE
            keep.append(o)
            vis.append(VisObj(o.index, kind, o.x, o.y, self.radius, o.time - t))
        self.live = keep
        return sort_visible(vis)

    def _ball(self) -> Optional[VisBall]:
        t = self.t
        for o in self.sliders:
            if o.time <= t <= o.end_time:
                b = o.ball_pos(t)
                b2 = o.ball_pos(t + 0.01)
                v = (b2 - b) / 0.01
                return VisBall(float(b[0]), float(b[1]), float(v[0]), float(v[1]))
        return None

    def _observe(self, vis_true: Optional[list] = None) -> np.ndarray:
        if vis_true is None:
            vis_true = self._visible_true()
        vis = vis_true
        if self.noise:
            nt, npx = self.pc.ttl_noise, self.pc.pos_noise
            vis = []
            for o in vis_true:
                b = self.bias.get(o.id)
                if b is None:
                    b = self.rng.normal(0, 1, 3) * np.array([npx * 0.6, npx * 0.6, nt * 0.7])
                    self.bias[o.id] = b
                j = self.rng.normal(0, 1, 3) * np.array([npx * 0.3, npx * 0.3, nt * 0.5])
                vis.append(VisObj(o.id, o.kind, o.x + b[0] + j[0], o.y + b[1] + j[1], o.r, o.ttl + b[2] + j[2]))
            vis = sort_visible(vis)
        return encode(self.pc.n_objs, self.pos, self.vel, self.keys, self.key_age, vis, self._ball(), self.preempt)
