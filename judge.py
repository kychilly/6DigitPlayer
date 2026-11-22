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


JUDGE_VERSION = "stable-2"   # bump when judging rules change -> cached replay scores are recomputed
MISS_WINDOW = 0.400     # osu!: clicking ON a circle earlier than the 50 window (but within 400 ms) is a MISS


def grade_for(dt: float, od: float) -> str:
    """osu!stable's exact comparison: abs(round(hit error in ms)) < floor(window in ms)."""
    e = abs(round(dt * 1000.0))
    w300, w100, w50 = (math.floor(w * 1000.0) for w in hit_windows(od))
    return "300" if e < w300 else "100" if e < w100 else "50" if e < w50 else "miss"


@dataclass
class Event:
    kind: str            # "300" | "100" | "50" | "miss" | "stray" | "head" (slider head hit; slider graded at its end)
    obj: int = -1
    dt: float = 0.0      # tap time minus ideal time (s); + = late
    t: float = 0.0


def spinner_required_revs(duration: float, od: float) -> float:
    sps = 3 + 0.4 * od if od <= 5 else 5 + 0.5 * (od - 5)
    return max(0.5, 0.6 * duration * sps)


def slider_checks(o: HitObject) -> list:
    """Times osu!stable checks a slider after its head: ticks, repeats, and the 'legacy last tick' 36 ms before
    the end. A check passes if a key is held and the cursor is on the follow circle."""
    out = []
    slides = max(1, int(o.slides))
    span = (o.end_time - o.time) / slides
    ti = float(getattr(o, "tick_interval", 0.0) or 0.0)
    for s in range(slides):
        s0 = o.time + s * span
        if ti > 1e-3:
            k = 1
            while s0 + k * ti < s0 + span - 0.010:
                out.append(s0 + k * ti)
                k += 1
        if s < slides - 1:
            out.append(s0 + span)                                     # repeat arrow
    out.append(max(o.time + (o.end_time - o.time) / 2.0, o.end_time - 0.036))   # end
    return sorted(out)


class Judge:
    """Incremental judge over TRUE objects (simulator / replay), following osu!stable's rules:
      * circles: graded by tap timing (300/100/50); clicking ON a circle up to 400 ms too early = MISS;
        clicking off the circle, or even earlier, is ignored (counted as a stray click);
      * sliders: head + every tick / repeat / end count as parts -> all parts = 300, at least half = 100,
        any = 50, none = miss (graded when the slider ends; head timing doesn't change the grade);
      * spinners: enough spins = 300, >90% = 100, >75% = 50, else miss."""

    FOLLOW_R = 2.0      # fitted to osu!stable results on real replays (2.4 = the follow-circle sprite, too lenient)

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
        self._checks = {o.index: slider_checks(o) for o in self.special if o.kind == SLIDER}
        self._parts = {o.index: [0, 0] for o in self.special if o.kind == SLIDER}       # [hit, done] after head
        self._head_ok: dict[int, bool] = {}
        self._tracking: dict[int, bool] = {}                     # slider follow state (osu!: 1x radius to start, 2.4x to keep)
        self._finished: set = set()
        self._finish_after_head: set = set()
        self._pending: list = []                                 # slider / spinner results, returned by advance()
        self._prev_ang: Optional[float] = None
        self.counts = {"300": 0, "100": 0, "50": 0, "miss": 0, "stray": 0}

    # ------------------------------------------------------------------ taps
    def advance(self, t: float) -> list:
        ev = []
        while self.i < len(self.q) and t > self.q[self.i].time + self.w50:
            o = self.q[self.i]
            self._tap_result(o, None, 0.0, o.time + self.w50, ev, t)
            self.i += 1
        ev += self._pending
        self._pending = []
        return ev

    def press(self, t: float, pos) -> list:
        ev = self.advance(t)
        if self.i >= len(self.q):
            self.counts["stray"] += 1
            return ev + [Event("stray", -1, 0.0, t)]
        o = self.q[self.i]
        dt = t - o.time
        if dt < -MISS_WINDOW or math.hypot(pos[0] - o.x, pos[1] - o.y) > self.r:
            self.counts["stray"] += 1                            # far too early, or not on the circle: ignored
            return ev + [Event("stray", -1, dt, t)]
        g = grade_for(dt, self.od)
        if dt < 0 and g == "miss":                               # on the circle but too early: osu! counts a MISS
            g = None
        self._tap_result(o, g, dt, t, ev, t)
        self.i += 1
        return ev

    def _tap_result(self, o: HitObject, grade, dt: float, t_res: float, ev: list, t_now: float):
        """grade None = missed (not clicked in time, or clicked too early)."""
        self.resolved[o.index] = t_res
        self.dts[o.index] = dt
        if o.kind == SLIDER:                                     # head is one part of the slider's grade
            self._head_ok[o.index] = grade is not None
            ev.append(Event("head" if grade is not None else "head_miss", o.index, dt, t_now))
            if o.index in self._finish_after_head:
                self._finish(o, t_now)
            return
        g = grade or "miss"
        self.grades[o.index] = g
        self.counts[g] += 1
        ev.append(Event(g, o.index, dt, t_now))

    def _finish(self, o: HitObject, t: float):
        if o.index in self._finished:
            return
        if o.kind == SLIDER:
            if o.index not in self._head_ok:                     # head not judged yet (very short slider)
                self._finish_after_head.add(o.index)
                return
            hit, done = self._parts[o.index]
            remaining = len(self._checks[o.index]) - done         # checks we never got to see count as missed
            total = 1 + done + remaining
            got = hit + (1 if self._head_ok[o.index] else 0)
            frac = got / total
            g = "300" if got == total else "100" if frac >= 0.5 else "50" if got > 0 else "miss"
        else:
            need = spinner_required_revs(o.end_time - o.time, self.od)
            frac = self.revs[o.index] / max(need, 1e-6)
            g = "300" if frac >= 1.0 else "100" if frac > 0.9 else "50" if frac > 0.75 else "miss"
        self._finished.add(o.index)
        self.grades[o.index] = g
        self.counts[g] += 1
        self._pending.append(Event(g, o.index, 0.0, t))

    # --------------------------------------------------------------- sliders / spinners
    def track(self, t: float, dt: float, pos, key_down: bool):
        """-> (slider_follow_ok or None, spinner_revs_gained, spinner_obj_or_None)."""
        sp = self.special
        while self.j < len(sp) and sp[self.j].end_time < t:
            o = sp[self.j]
            if o.kind == SLIDER:                                 # checks between the last call and the end
                self._run_checks(o, o.end_time, pos, key_down)
            self._finish(o, t)
            self.j += 1
            self._prev_ang = None
        if self.j >= len(sp) or sp[self.j].time > t:
            self._prev_ang = None
            return None, 0.0, None
        o = sp[self.j]
        if o.kind == SLIDER:
            self._run_checks(o, t, pos, key_down)
            b = o.ball_pos(t)
            ok = self._update_tracking(o, b, pos, key_down)
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

    def _update_tracking(self, o: HitObject, b, pos, key_down: bool) -> bool:
        d = math.hypot(pos[0] - b[0], pos[1] - b[1])
        was = self._tracking.get(o.index, False)
        ok = bool(key_down and d <= (self.FOLLOW_R if was else 1.0) * self.r)
        self._tracking[o.index] = ok
        return ok

    def _run_checks(self, o: HitObject, t: float, pos, key_down: bool):
        checks, parts = self._checks[o.index], self._parts[o.index]
        while parts[1] < len(checks) and checks[parts[1]] <= t:
            b = o.ball_pos(checks[parts[1]])
            if self._update_tracking(o, b, pos, key_down):
                parts[0] += 1
            parts[1] += 1

    def finalize(self, t: float = 1e9) -> list:
        """End of play: grade every slider / spinner not graded yet (unseen slider checks count as missed)."""
        ev = self.advance(t)
        for o in self.special:
            if o.index not in self._finished:
                self._head_ok.setdefault(o.index, False) if o.kind == SLIDER else None
                self._finish(o, t)
        return ev + self._drain()

    def _drain(self) -> list:
        ev, self._pending = self._pending, []
        return ev

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
            elif e.kind == "head":                             # slider head hit (the slider is graded at its end)
                r += rc.slider_head
            if e.kind in ("300", "100", "50", "head") and e.dt != 0.0:
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
        if o.ttl > MISS_WINDOW:
            self.counts["stray"] += 1
            return Event("stray", -1, dt, now), None
        if math.hypot(cursor_pf[0] - o.x, cursor_pf[1] - o.y) <= o.r:
            g = grade_for(dt, self.od)                        # "miss" here = on the circle but too early
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
