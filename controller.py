"""High-rate control loop: turns tracked objects into human-like cursor motion + taps."""
from __future__ import annotations
import math
import time
import traceback
import threading
from dataclasses import dataclass
from enum import Enum
import numpy as np

from config import BotConfig, SLIDER_HEAD, SPINNER
from humanizer import TremorNoise, Trajectory, TapTimingModel, sample_aim_point
from inputs import enable_high_res_timer, check_foreground


class Mode(Enum):
    IDLE = "IDLE"
    SLIDER = "SLIDER"
    SPINNER = "SPINNER"


@dataclass
class Plan:
    obj_id: int
    aim: np.ndarray
    is_miss: bool
    tap_offset: float
    lead: float
    arrival: float
    stream: bool
    gap_next: float


class Controller(threading.Thread):
    def __init__(self, cfg: BotConfig, tracker, audio, backend, shared):
        super().__init__(daemon=True)
        self.cfg, self.h = cfg, cfg.human
        self.tracker, self.audio, self.backend, self.shared = tracker, audio, backend, shared
        self.rng = np.random.default_rng()
        self.scale = cfg.capture.height / 1080.0
        self.origin = np.array([cfg.capture.left, cfg.capture.top], float)
        self.timing = TapTimingModel(cfg.game.od, self.h, self.rng)
        self.tremor = TremorNoise(self.rng, self.h.tremor_amp_px * self.scale, self.h.tremor_drift_px * self.scale)
        start = backend.get_pos() or (cfg.capture.left + cfg.capture.width / 2, cfg.capture.top + cfg.capture.height / 2)
        self.pos = np.array(start, float)
        self.mode = Mode.IDLE
        self.traj: Trajectory | None = None
        self.plan: Plan | None = None
        self.key_idx = 0
        self.last_tap_time = -1e9
        self.last_hit_nominal: float | None = None
        self.releases: dict[str, float] = {}
        # slider
        self.slider_key = None
        self.slider_start = 0.0
        self.last_ball_seen = 0.0
        self.follow_off = np.zeros(2)
        # spinner
        self.spin_id = None
        self.spin_center = np.zeros(2)
        self.spin_angle = 0.0
        self.spin_rad = 0.0
        self.spin_R = 100.0
        self.spin_omega = 30.0
        self.last_spin_seen = 0.0
        self._stop_flag = False
        # control gate: bot only touches mouse/keys while engaged
        self.engaged = False
        self._seen_since = None
        self._last_visible = 0.0
        self._fg_checked = 0.0
        self._fg_ok = False

    def stop(self):
        self._stop_flag = True

    # ------------------------------------------------------------------ loop
    def run(self):
        enable_high_res_timer()
        last = time.perf_counter()
        while not self._stop_flag:
            now = time.perf_counter()
            try:
                self.step(now, max(1e-4, now - last))
            except Exception:
                traceback.print_exc()
            last = now
            time.sleep(1.0 / self.cfg.control_hz * 0.5)
        self._release_all()

    def step(self, now: float, dt: float):
        armed = self.shared.armed
        for k in [k for k, t in self.releases.items() if now >= t]:
            del self.releases[k]
            self.backend.key_up(k)            # always: never leave a key stuck down
        if not self._update_gate(now, armed):
            with self.shared.lock:
                self.shared.mode = "STANDBY" if armed else "DISARMED"
            return
        if self.mode is Mode.SLIDER:
            base = self._slider_step(now, dt)
        elif self.mode is Mode.SPINNER:
            base = self._spinner_step(now, dt)
        else:
            base = self._move_step(now)
        self.pos = base
        out = base + self.tremor(now)
        if armed:
            self.backend.move(out[0], out[1])
        with self.shared.lock:
            self.shared.cursor = (float(out[0]), float(out[1]))
            self.shared.mode = self.mode.value

    # ------------------------------------------------------------------ control gate
    def _osu_focused(self, now: float) -> bool:
        g = self.cfg.gate
        if not g.require_focus:
            return True
        if now - self._fg_checked > 0.05:           # poll the OS at 20 Hz, not 1 kHz
            self._fg_checked = now
            self._fg_ok, proc, title = check_foreground(g.process_prefix, g.title_gate)
            self.shared.fg = (proc, title)
        return self._fg_ok

    def _update_gate(self, now: float, armed: bool) -> bool:
        """True only while armed AND osu! is focused AND a map's objects are on screen."""
        g = self.cfg.gate
        if not armed:
            if self.engaged:
                self._disengage("disarmed")
            self._seen_since = None
            return False
        focused = self._osu_focused(now)
        ball = self.tracker.ball()
        if self.engaged:   # stay engaged while any note / slider / spinner is around
            visible = (self.tracker.has_targets() or self.mode is not Mode.IDLE
                       or (ball is not None and now - ball.t < 0.3))
        else:              # to START, require proof it's a real map: a note with a steady hit time
            visible = self.tracker.has_verified()
        if focused and visible:
            self._last_visible = now
            if self._seen_since is None:
                self._seen_since = now
            if not self.engaged and now - self._seen_since >= g.engage_after:
                self.engaged = True
                self.pos = np.array(self.backend.get_pos() or self.pos, float)  # start from real cursor
                self.shared.log("ENGAGED: map detected, taking control")
        else:
            self._seen_since = None
        if self.engaged:
            if not focused:
                self._disengage("osu! lost focus")
            elif now - self._last_visible > g.idle_release:
                self._disengage("no objects on screen")
        return self.engaged

    def _disengage(self, reason: str):
        self.engaged = False
        self._release_all()
        self.releases.clear()
        self.slider_key = None
        self.mode, self.traj, self.plan = Mode.IDLE, None, None
        self.shared.log(f"STANDBY: {reason}")

    def _release_all(self):
        for k in list(self.releases) + ([self.slider_key] if self.slider_key else []):
            self.backend.key_up(k)

    # ------------------------------------------------------------------ helpers
    def _abs(self, obj) -> np.ndarray:
        return np.array([obj.x, obj.y]) + self.origin

    def _refined_hit(self, obj) -> float:
        if self.audio is None:
            return obj.hit_time
        return self.audio.refine(obj.hit_time, self.h.audio_window, self.h.audio_weight)

    def _publish_path(self):
        if self.traj is not None:
            with self.shared.lock:
                self.shared.path = [tuple(p) for p in self.traj.sample_path(20)]
                self.shared.aim_point = tuple(self.traj.p1)

    # ------------------------------------------------------------------ tap/move mode
    def _move_step(self, now: float) -> np.ndarray:
        objs = self.tracker.pending()
        if objs:
            obj = objs[0]
            if obj.cls == SPINNER:
                self._enter_spinner(obj, now)
                return self.pos
            if self.plan is None or self.plan.obj_id != obj.id:
                self._make_plan(objs, now)
            plan = self.plan
            tap_time = self._refined_hit(obj) + plan.tap_offset - self.h.latency_comp
            arrival = tap_time - plan.lead
            if abs(arrival - plan.arrival) > 0.04 and arrival - now > 0.06:
                self._replan(arrival, now)
            if now >= tap_time:
                self._tap(obj, now, plan)
        return self.traj.position(now) if self.traj else self.pos

    def _make_plan(self, objs, now: float):
        h, rng = self.h, self.rng
        obj = objs[0]
        nxt = objs[1] if len(objs) > 1 and objs[1].cls != SPINNER else None
        est = self._refined_hit(obj)
        gap_prev = est - self.last_hit_nominal if self.last_hit_nominal else math.inf
        gap_next = (nxt.hit_time - obj.hit_time) if nxt else math.inf
        stream = min(gap_prev, gap_next) < self.cfg.game.stream_gap
        tap_off = self.timing.sample(stream)
        lead = max(0.0, rng.normal(h.lead_mean, h.lead_std))
        arrival = est + tap_off - h.latency_comp - lead

        center = self._abs(obj)
        r = max(obj.radius, 8 * self.scale)
        dist = float(np.linalg.norm(center - self.pos))
        avail = arrival - now
        fitts = (h.fitts_a + h.fitts_b * math.log2(dist / (2 * r) + 1)) * h.speed_factor * rng.uniform(0.9, 1.15)
        dur = max(h.min_move_time, min(avail, max(fitts, 0.65 * avail)))
        aim, miss = sample_aim_point(center, r, dist, dur, rng, h)

        v0 = self.traj.velocity(now) if self.traj else np.zeros(2)
        v1 = None
        if nxt is not None and gap_next < 0.30:   # keep momentum toward the next note
            d = self._abs(nxt) - aim
            dn = float(np.linalg.norm(d))
            if dn > 1e-3:
                v1 = d / dn * min(0.45 * dn / max(gap_next, 0.06), 1.2 * dist / max(dur, 0.03))
        curve = abs(rng.normal(h.curve_mean, h.curve_std))
        self.traj = Trajectory(self.pos, v0, aim, now, dur, rng, curve, v1)
        self.plan = Plan(obj.id, aim, miss, tap_off, lead, arrival, stream, gap_next)
        self._publish_path()

    def _replan(self, arrival: float, now: float):
        p = self.plan
        T = max(self.h.min_move_time, arrival - now)
        v0 = self.traj.velocity(now) if self.traj else np.zeros(2)
        cur = self.traj.position(now) if self.traj else self.pos
        self.traj = Trajectory(cur, v0, p.aim, now, T, self.rng, abs(self.rng.normal(0.02, 0.01)))
        p.arrival = arrival
        self._publish_path()

    def _tap(self, obj, now: float, plan: Plan):
        gs = self.cfg.game
        if plan.stream and (now - self.last_tap_time) < 0.30:
            self.key_idx = 1 - self.key_idx          # alternate in streams
        else:
            self.key_idx = 0                          # single-tap with primary key
        key = gs.keys[self.key_idx]
        armed = self.shared.armed
        if key in self.releases:                      # same key still down -> lift first
            del self.releases[key]
            if armed:
                self.backend.key_up(key)
        if armed:
            self.backend.key_down(key)
        if obj.cls == SLIDER_HEAD:
            self.mode, self.slider_key = Mode.SLIDER, key
            self.slider_start = self.last_ball_seen = now
            self.follow_off[:] = 0
        else:
            hold = self.timing.hold_time()
            if math.isfinite(plan.gap_next):
                hold = min(hold, 0.6 * plan.gap_next)
            self.releases[key] = now + hold
        self.tracker.mark_clicked(obj.id, now)        # forget it
        self.last_tap_time, self.last_hit_nominal = now, obj.hit_time
        self.plan = None
        self.shared.log(f"tap #{obj.id} {'head' if obj.cls == SLIDER_HEAD else 'circle'} "
                        f"key={key} off={plan.tap_offset * 1000:+.0f}ms{' AIM-MISS' if plan.is_miss else ''}")

    # ------------------------------------------------------------------ slider follow
    def _slider_step(self, now: float, dt: float) -> np.ndarray:
        h = self.h
        ball = self.tracker.ball()
        pos = self.traj.position(now) if (self.traj and not self.traj.finished(now)) else self.pos
        if ball is not None and ball.t > self.slider_start - 0.1:
            self.last_ball_seen = max(self.last_ball_seen, ball.t) if ball.t >= self.last_ball_seen else self.last_ball_seen
            lead = min(0.15, max(0.0, now - ball.t))
            target = np.array([ball.x + ball.vx * lead, ball.y + ball.vy * lead]) + self.origin
            self.follow_off += (-3.0 * self.follow_off * dt
                                + h.follow_noise_px * self.scale * math.sqrt(dt) * 6 * self.rng.normal(size=2))
            a = 1 - math.exp(-h.follow_rate * dt)
            pos = pos + (target + self.follow_off - pos) * a
            self.traj = None
        if now - self.slider_start > 0.35 and now - self.last_ball_seen > 0.12:
            if self.shared.armed and self.slider_key:
                self.backend.key_up(self.slider_key)
            self.slider_key, self.mode = None, Mode.IDLE
            self.shared.log("slider end")
        return pos

    # ------------------------------------------------------------------ spinner
    def _enter_spinner(self, obj, now: float):
        self.tracker.mark_active(obj.id)
        self.mode, self.spin_id = Mode.SPINNER, obj.id
        self.spin_center = self._abs(obj)
        rel = self.pos - self.spin_center
        self.spin_angle = math.atan2(rel[1], rel[0])
        self.spin_rad = float(np.linalg.norm(rel))
        self.spin_R = float(np.clip(0.5 * obj.radius, 30 * self.scale, 160 * self.scale))
        self.spin_omega = self.rng.normal(self.h.spin_omega_mean, self.h.spin_omega_std) * self.rng.choice([-1, 1])
        self.last_spin_seen = now
        self.traj = None
        self.shared.log("spinner start")

    def _spinner_step(self, now: float, dt: float) -> np.ndarray:
        obj = self.tracker.get(self.spin_id)
        if obj is not None:
            self.last_spin_seen = now
            self.spin_center += 0.2 * (self._abs(obj) - self.spin_center)
        elif now - self.last_spin_seen > 0.2:
            self.mode = Mode.IDLE
            self.shared.log("spinner end")
            return self.pos
        self.spin_omega += self.rng.normal(0, 4.0) * dt   # speed wobble
        self.spin_angle += self.spin_omega * dt
        self.spin_rad += (self.spin_R - self.spin_rad) * (1 - math.exp(-8 * dt))
        return self.spin_center + self.spin_rad * np.array([math.cos(self.spin_angle), math.sin(self.spin_angle)])
