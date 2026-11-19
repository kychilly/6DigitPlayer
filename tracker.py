"""Short-term object memory. Holds ONLY what is currently visible on screen.
Clicked objects are deleted immediately (and forgotten); no beatmap is ever stored."""
from __future__ import annotations
import math
import threading
from dataclasses import dataclass, field
from typing import Optional
from config import CIRCLE, SLIDER_HEAD, SPINNER, SLIDER_BALL, GameConfig, TrackerConfig, ar_to_preempt


@dataclass
class TrackedObject:
    id: int
    cls: int
    x: float
    y: float
    radius: float
    hit_time: float          # absolute perf_counter seconds (estimated)
    first_seen: float
    last_seen: float
    hits: int = 1
    votes: dict = field(default_factory=lambda: {CIRCLE: 0.0, SLIDER_HEAD: 0.0, SPINNER: 0.0})
    active: bool = False     # spinner currently being handled by the controller
    est_first: float = 0.0   # first raw hit-time estimate (for the stationarity check)
    verified: bool = False   # hit time held steady -> behaves like a real note
    bad: bool = False        # hit time drifts with the clock -> static thing, not a note


@dataclass
class BallState:
    x: float
    y: float
    t: float
    vx: float = 0.0
    vy: float = 0.0


class ObjectTracker:
    def __init__(self, game: GameConfig, cfg: TrackerConfig):
        self.cfg = cfg
        self.preempt = ar_to_preempt(game.ar)
        self._objs: dict[int, TrackedObject] = {}
        self._next = 1
        self._suppress = []  # (x, y, r, expires)
        self._ball: Optional[BallState] = None
        self.lock = threading.RLock()

    def set_preempt(self, preempt: float):
        """New map with a different AR: approach-ring progress now maps to a different time-to-hit.
        Objects tracked under the old AR are dropped (their hit-time estimates would be wrong)."""
        with self.lock:
            if abs(preempt - self.preempt) > 1e-6:
                self.preempt = preempt
                self._objs.clear()

    # ------------------------------------------------------------------ perception side
    def update(self, dets, t_frame: float, scale: float, frame_h: float = None):
        """dets are in model-input pixels; `scale` converts them to frame pixels."""
        c = self.cfg
        with self.lock:
            self._suppress = [s for s in self._suppress if s[3] > t_frame]
            time_gate = 0.05 + 0.06 * self.preempt
            used = set()
            for d in sorted(dets, key=lambda d: -d.score):
                x, y, r = d.x * scale, d.y * scale, d.radius * scale
                if frame_h and d.cls != SPINNER and not (c.min_radius_frac * frame_h <= r <= c.max_radius_frac * frame_h):
                    continue  # implausible circle size (menu logo, UI blobs)
                if d.cls == SLIDER_BALL:
                    self._update_ball(x, y, t_frame)
                    continue
                if d.progress < 0.25 and any(math.hypot(x - sx, y - sy) < sr
                                             for sx, sy, sr, _ in self._suppress):
                    continue  # ghost of an object we just clicked
                est = t_frame + d.progress * self.preempt
                best, bd = None, 1e9
                for o in self._objs.values():
                    if o.id in used:
                        continue
                    if (o.cls == SPINNER) != (d.cls == SPINNER):
                        continue
                    dist = math.hypot(x - o.x, y - o.y)
                    if dist < max(0.8 * o.radius, 10) and abs(est - o.hit_time) < time_gate and dist < bd:
                        best, bd = o, dist
                if best is None:
                    o = TrackedObject(self._next, d.cls, x, y, r, est, t_frame, t_frame)
                    self._next += 1
                    o.votes[d.cls] += d.score
                    o.est_first = est
                    self._objs[o.id] = o
                    used.add(o.id)
                else:
                    o = best
                    o.x += c.ema_pos * (x - o.x)
                    o.y += c.ema_pos * (y - o.y)
                    o.radius += 0.3 * (r - o.radius)
                    o.hit_time += c.ema_time * (est - o.hit_time)
                    o.last_seen = t_frame
                    o.hits += 1
                    o.votes[d.cls] += d.score
                    o.cls = max(o.votes, key=o.votes.get)
                    if o.cls != SPINNER and t_frame - o.first_seen >= c.verify_age:
                        ok = abs(o.hit_time - o.est_first) <= c.verify_tol
                        if ok:
                            o.verified = True
                        o.bad = (not ok) and (not o.verified)
                    used.add(o.id)
            for oid in [i for i, o in self._objs.items() if t_frame - o.last_seen > c.forget_after]:
                del self._objs[oid]

    def _update_ball(self, x, y, t):
        b = self._ball
        if b is not None and t - b.t < 0.3 and t > b.t:
            dt = t - b.t
            vx, vy = (x - b.x) / dt, (y - b.y) / dt
            self._ball = BallState(x, y, t, 0.6 * b.vx + 0.4 * vx, 0.6 * b.vy + 0.4 * vy)
        else:
            self._ball = BallState(x, y, t)

    # ------------------------------------------------------------------ control side
    def pending(self):
        """Confirmed, un-clicked tap objects, soonest first."""
        with self.lock:
            out = [o for o in self._objs.values() if o.hits >= self.cfg.min_hits and not o.active and not o.bad]
            out.sort(key=lambda o: o.hit_time)
            return [TrackedObject(**{**o.__dict__}) for o in out]

    def has_targets(self) -> bool:
        """Cheap check: is any confirmed object (incl. an active spinner) on screen?"""
        with self.lock:
            return any(o.hits >= self.cfg.min_hits and not o.bad for o in self._objs.values())

    def has_verified(self) -> bool:
        """Evidence a real map is being played: a tracked note whose hit time held steady."""
        with self.lock:
            return any(o.verified and not o.bad and o.cls != SPINNER for o in self._objs.values())

    def get(self, oid: int) -> Optional[TrackedObject]:
        with self.lock:
            o = self._objs.get(oid)
            return TrackedObject(**{**o.__dict__}) if o else None

    def mark_active(self, oid: int):
        with self.lock:
            if oid in self._objs:
                self._objs[oid].active = True

    def mark_clicked(self, oid: int, now: float):
        """Forget a clicked object and suppress its fade-out ghost for a moment."""
        with self.lock:
            o = self._objs.pop(oid, None)
            if o is not None:
                self._suppress.append((o.x, o.y, o.radius * 1.4, now + self.cfg.suppress_time))

    def ball(self) -> Optional[BallState]:
        with self.lock:
            b = self._ball
            return BallState(b.x, b.y, b.t, b.vx, b.vy) if b else None

    def snapshot(self):
        with self.lock:
            return sorted((TrackedObject(**{**o.__dict__}) for o in self._objs.values()),
                          key=lambda o: o.hit_time)