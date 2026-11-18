"""osu! judgement rules + the reward function. Used by three things:
  * the simulator (judges the policy's own play, exact),
  * the replay dataset builder (judges a human replay so we know which notes were hit, when),
  * the live bot (LiveJudge: judges taps against what the screen tracker sees).
The reward is identical in all three, so sim-trained behaviour transfers.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import RewardConfig
from osu_formats import CIRCLE, SLIDER, SPINNER, HitObject


def hit_windows(od: float):
    """(w300, w100, w50) half-widths in seconds."""
    return (80 - 6 * od) / 1000.0, (140 - 8 * od) / 1000.0, (200 - 10 * od) / 1000.0


def grade_for(dt: float, od: float) -> str:
    w300, w100, w50 = hit_windows(od)
    a = abs(dt)
    return "300" if a <= w300 else "100" if a <= w100 else "50" if a <= w50 else "miss"


@dataclass
class Event:
    kind: str            # "300" | "100" | "50" | "miss" | "stray"
    obj: int = -1
    dt: float = 0.0      # tap time minus ideal time (s); + = late
    t: float = 0.0


def spinner_required_revs(duration: float, od: float) -> float:
    sps = 3 + 0.4 * od if od <= 5 else 5 + 0.5 * (od - 5)
    return max(0.5, 0.6 * duration * sps)


class Judge:
    """Incremental judge over TRUE objects (simulator / replay)."""

    FOLLOW_R = 2.4

    def __init__(self, objects: list, od: float, radius: float):
        self.od, self.r = od, radius
        self.w300, self.w100, self.w50 = hit_windows(od)
        self.q = [o for o in objects if o.kind != SPINNER]       # tap queue (circles + slider heads)
        self.i = 0
        self.special = [o for o in objects if o.kind != CIRCLE]  # sliders + spinners
        self.j = 0
        self.resolved: dict[int, float] = {}                     # obj.index -> time it left the screen queue
        self.grades: dict[int, str] = {}
        self.dts: dict[int, float] = {}
        self.follow = {o.index: [0.0, 0.0] for o in self.special if o.kind == SLIDER}   # [follow s, total s]
        self.revs = {o.index: 0.0 for o in self.special if o.kind == SPINNER}
        self._prev_ang: Optional[float] = None
        self.counts = {"300": 0, "100": 0, "50": 0, "miss": 0, "stray": 0}

    # ------------------------------------------------------------------ taps
    def advance(self, t: float) -> list:
        ev = []
        while self.i < len(self.q) and t > self.q[self.i].time + self.w50:
            o = self.q[self.i]
            self._resolve(o, "miss", 0.0, o.time + self.w50)
            ev.append(Event("miss", o.index, 0.0, t))
            self.i += 1
        return ev

    def press(self, t: float, pos) -> list:
        ev = self.advance(t)
        if self.i >= len(self.q):
            self.counts["stray"] += 1
            return ev + [Event("stray", -1, 0.0, t)]
        o = self.q[self.i]
        dt = t - o.time
        if dt < -self.w50:                                    # way too early: ignored by the game
            self.counts["stray"] += 1
            return ev + [Event("stray", -1, dt, t)]
        if math.hypot(pos[0] - o.x, pos[1] - o.y) <= self.r:
            g = grade_for(dt, self.od)
            self._resolve(o, g, dt, t)
            self.i += 1
            return ev + [Event(g, o.index, dt, t)]
        self.counts["stray"] += 1                             # click outside the circle: ignored
        return ev + [Event("stray", -1, dt, t)]

    def _resolve(self, o: HitObject, grade: str, dt: float, t: float):
        self.resolved[o.index] = t
        self.grades[o.index] = grade
        self.dts[o.index] = dt
        self.counts[grade] += 1

    # --------------------------------------------------------------- sliders / spinners
    def track(self, t: float, dt: float, pos, key_down: bool):
        """-> (slider_follow_ok or None, spinner_revs_gained, spinner_obj_or_None)."""
        sp = self.special
        while self.j < len(sp) and sp[self.j].end_time < t:
            self.j += 1
            self._prev_ang = None
        if self.j >= len(sp) or sp[self.j].time > t:
            self._prev_ang = None
            return None, 0.0, None
        o = sp[self.j]
        if o.kind == SLIDER:
            b = o.ball_pos(t)
            ok = bool(key_down and math.hypot(pos[0] - b[0], pos[1] - b[1]) <= self.FOLLOW_R * self.r)
            f = self.follow[o.index]
            f[1] += dt
            f[0] += dt if ok else 0.0
            return ok, 0.0, None
        ang = math.atan2(pos[1] - 192.0, pos[0] - 256.0)
        d = 0.0
        if self._prev_ang is not None:
            d = (ang - self._prev_ang + math.pi) % (2 * math.pi) - math.pi
        self._prev_ang = ang
        rev = abs(d) / (2 * math.pi) if math.hypot(pos[0] - 256, pos[1] - 192) > 20 else 0.0
        self.revs[o.index] += rev
        return None, rev, o

    def accuracy(self) -> float:
        c = self.counts
        n = c["300"] + c["100"] + c["50"] + c["miss"]
        return (300 * c["300"] + 100 * c["100"] + 50 * c["50"]) / (300.0 * n) if n else 1.0


# ====================================================================== reward
class RewardShaper:
    """Turns judgement events + cursor state into a scalar reward for one step.
    `objs` are VisObj (what the policy sees), so the same code runs live and in the simulator."""

    def __init__(self, rc: RewardConfig, od: float):
        self.rc, self.od = rc, od
        self.w300, self.w100, self.w50 = hit_windows(od)
        self._target_id: Optional[int] = None
        self._target_dist = 0.0
        self._spin_budget: dict[int, float] = {}

    def reset(self):
        self._target_id, self._spin_budget = None, {}

    def step(self, events, cursor, objs, keys_down, follow_ok: Optional[bool],
             spin_revs: float, spinner_id: Optional[int] = None, spinner_need: float = 0.0) -> float:
        rc, r = self.rc, 0.0
        for e in events:
            if e.kind == "300":
                r += rc.r300
            elif e.kind == "100":
                r += rc.r100
            elif e.kind == "50":
                r += rc.r50
            elif e.kind == "miss":
                r += rc.miss
            elif e.kind == "stray":
                r += rc.stray_click
            if e.kind in ("300", "100", "50"):
                r += rc.timing_bonus * math.exp(-(e.dt / rc.timing_sigma) ** 2)

        taps = [o for o in objs if o.kind != SPINNER]
        tgt = min(taps, key=lambda o: o.ttl) if taps else None
        if tgt is not None and tgt.ttl < 0.6:
            d = math.hypot(cursor[0] - tgt.x, cursor[1] - tgt.y)
            if self._target_id == tgt.id:                      # progress toward the SAME note only
                r += rc.approach_shaping * (self._target_dist - d) / 256.0
            self._target_id, self._target_dist = tgt.id, d
            if -self.w50 <= tgt.ttl <= 0.1 and d <= tgt.r:
                r += rc.on_target
        else:
            self._target_id = None

        if follow_ok is not None and follow_ok:
            r += rc.slider_follow
        if spin_revs > 0 and spinner_id is not None:
            left = self._spin_budget.setdefault(spinner_id, spinner_need)
            use = min(spin_revs, left)
            self._spin_budget[spinner_id] = left - use
            r += rc.spin_per_rev * use
        if (keys_down[0] or keys_down[1]) and not objs and follow_ok is None and spinner_id is None:
            r -= rc.idle_key_penalty
        return r


# ====================================================================== live judge
class LiveJudge:
    """Judges taps against the tracker's visible objects (no map). Mirrors Judge.press / advance."""

    def __init__(self, od: float):
        self.od = od
        self.w300, self.w100, self.w50 = hit_windows(od)
        self.counts = {"300": 0, "100": 0, "50": 0, "miss": 0, "stray": 0}

    def reset(self):
        self.counts = {k: 0 for k in self.counts}

    def press(self, now: float, cursor_pf, objs_pf: list) -> tuple:
        """objs_pf: VisObj list in playfield coords sorted by ttl. -> (Event, hit VisObj or None)."""
        taps = [o for o in objs_pf if o.kind != SPINNER]
        if not taps:
            self.counts["stray"] += 1
            return Event("stray", -1, 0.0, now), None
        o = taps[0]
        dt = -o.ttl                                           # tap now; ttl>0 means early
        if o.ttl > self.w50:
            self.counts["stray"] += 1
            return Event("stray", -1, dt, now), None
        if math.hypot(cursor_pf[0] - o.x, cursor_pf[1] - o.y) <= o.r:
            g = grade_for(dt, self.od)
            self.counts[g] += 1
            return Event(g, o.id, dt, now), o
        self.counts["stray"] += 1
        return Event("stray", -1, dt, now), None

    def overdue(self, objs_pf: list) -> list:
        out = []
        for o in objs_pf:
            if o.kind != SPINNER and o.ttl < -self.w50:
                self.counts["miss"] += 1
                out.append(o)
        return out

    def accuracy(self) -> float:
        c = self.counts
        n = c["300"] + c["100"] + c["50"] + c["miss"]
        return (300 * c["300"] + 100 * c["100"] + 50 * c["50"]) / (300.0 * n) if n else 1.0
