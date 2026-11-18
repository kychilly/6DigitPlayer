"""Plays real osu! with the LEARNED policy (replaces the hand-written humanizer controller).

Every 1/policy.hz seconds it builds the same observation the simulator uses - from the screen
tracker only, never a map - asks the network for [aim velocity, key0, key1], and applies it.
At 1 kHz the cursor follows that velocity with a hand-acceleration limit plus physiological
tremor, so the motion stays smooth and human.

With a learner attached (--learn) it also JUDGES its own taps exactly like osu! would (300/100/50/
miss from the tracker's timing + cursor position), turns that into the reward in config.RewardConfig,
and PPO-updates the policy after every map.
"""
from __future__ import annotations
import math
import traceback

import numpy as np

from config import BotConfig, ar_to_preempt
from controller import Controller, Mode
from judge import LiveJudge, RewardShaper, Event
from live_learning import ExperienceBuffer
from obs import VisObj, VisBall, SPINNER, encode, sort_visible
from playfield import PlayfieldMap

SPINNER_REV_CAP = 10.0


class PolicyController(Controller):
    def __init__(self, cfg: BotConfig, tracker, audio, backend, shared, runner, learner=None):
        super().__init__(cfg, tracker, audio, backend, shared)
        self.runner, self.learner = runner, learner
        self.pc = cfg.policy
        self.pfm = PlayfieldMap(cfg.capture, cfg.playfield)
        self.period = 1.0 / self.pc.hz
        self.preempt = ar_to_preempt(cfg.game.ar)
        self.judge = LiveJudge(cfg.game.od)
        self.shaper = RewardShaper(cfg.reward, cfg.game.od)
        self.buf = ExperienceBuffer()
        self.pf = np.array([256.0, 192.0])
        self.v = np.zeros(2)
        self.v_des = np.zeros(2)
        self.keys = [False, False]
        self.key_t = [0.0, 0.0]
        self.next_decision = 0.0
        self.active = False
        self.circle_r = 32.0
        self._spin_ang = None
        self._n_steps = 0

    # ------------------------------------------------------------------ main step (1 kHz)
    def step(self, now: float, dt: float):
        armed = self.shared.armed
        if not self._update_gate(now, armed):
            with self.shared.lock:
                self.shared.mode = "STANDBY" if armed else "DISARMED"
            return
        if not self.active:
            self._begin_episode(now)
        if now >= self.next_decision:
            try:
                self._decide(now)
            except Exception:
                traceback.print_exc()
            self.next_decision = max(self.next_decision + self.period, now - self.period)
        # accel-limited velocity -> position (same dynamics as the simulator)
        dv = self.v_des - self.v
        lim = self.pc.max_accel * dt
        n = float(np.linalg.norm(dv))
        if n > lim:
            dv *= lim / n
        self.v = self.v + dv
        self.pf = np.clip(self.pf + self.v * dt, [-30.0, -20.0], [542.0, 404.0])
        sx, sy = self.pfm.pf_to_screen(*self.pf)
        out = np.array([sx, sy]) + self.tremor(now)
        if armed:
            self.backend.move(out[0], out[1])
        self.pos = np.array([sx, sy])
        with self.shared.lock:
            self.shared.cursor = (float(out[0]), float(out[1]))
            self.shared.mode = "POLICY"
            self.shared.path, self.shared.aim_point = [], None

    # ------------------------------------------------------------------ episode bookkeeping
    def _begin_episode(self, now: float):
        self.active = True
        sx, sy = self.backend.get_pos() or tuple(self.pos)
        self.pf = np.array(self.pfm.screen_to_pf(sx, sy), float)
        self.v = np.zeros(2); self.v_des = np.zeros(2)
        self.keys = [False, False]
        self.key_t = [now, now]
        self.next_decision = now
        self._spin_ang = None
        self.judge.reset(); self.shaper.reset()
        self.buf = ExperienceBuffer()
        self._n_steps = 0

    def _end_episode(self):
        if not self.active:
            return
        self.active = False
        c = dict(self.judge.counts)
        acc = self.judge.accuracy()
        self.shared.log(f"map finished: acc {acc * 100:.1f}% {c['300']}/{c['100']}/{c['50']}/{c['miss']} (300/100/50/miss)")
        if self.learner is not None:
            try:
                self.learner.finish_episode(self.buf, acc, c)
            except Exception:
                traceback.print_exc()
        self.buf = ExperienceBuffer()

    def _disengage(self, reason: str):
        super()._disengage(reason)
        self._end_episode()

    def _release_all(self):
        for i, k in enumerate(self.cfg.game.keys):
            if self.keys[i]:
                try:
                    self.backend.key_up(k)
                except Exception:
                    pass
        self.keys = [False, False]

    # ------------------------------------------------------------------ perception -> observation
    def _visible(self, now: float) -> list:
        s = self.pfm.s
        out = []
        for o in self.tracker.pending():
            x, y = self.pfm.frame_to_pf(o.x, o.y)
            if o.cls == SPINNER:
                r = self.circle_r
            else:
                r = max(o.radius / s, 6.0)
                self.circle_r += 0.02 * (r - self.circle_r)
            out.append(VisObj(o.id, o.cls, x, y, r, self._refined_hit(o) - now))
        return sort_visible(out)

    def _ball(self, now: float):
        b = self.tracker.ball()
        if b is None or now - b.t > 0.3:
            return None
        s = self.pfm.s
        x, y = self.pfm.frame_to_pf(b.x, b.y)
        return VisBall(x, y, b.vx / s, b.vy / s)

    # ------------------------------------------------------------------ one policy decision (120 Hz)
    def _decide(self, now: float):
        objs = self._visible(now)
        ball = self._ball(now)
        age = [now - self.key_t[0], now - self.key_t[1]]
        obs = encode(self.pc.n_objs, self.pf, self.v, self.keys, age, objs, ball, self.preempt)
        act, logp, val = self.runner.act(obs)
        self.v_des = act[:2] * self.pc.max_speed
        new_keys = [bool(act[2] > 0.5), bool(act[3] > 0.5)]

        # --- apply keys
        rising = []
        for i, k in enumerate(self.cfg.game.keys):
            if new_keys[i] != self.keys[i]:
                self.key_t[i] = now
                if self.shared.armed:
                    (self.backend.key_down if new_keys[i] else self.backend.key_up)(k)
                if new_keys[i]:
                    rising.append(i)
        self.keys = new_keys

        # --- judge what happened (exactly osu!'s rules, from what we can see)
        events, cursor = [], self.pf + self.tremor(now) / self.pfm.s
        for o in self.judge.overdue(objs):                       # notes that slipped past the 50 window
            events.append(Event("miss", o.id, 0.0, now))
            self.tracker.mark_clicked(o.id, now)
            self.shared.log(f"MISS #{o.id}")
        objs = [o for o in objs if not (o.kind != SPINNER and o.ttl < -self.judge.w50)]
        if rising:
            ev, hit = self.judge.press(now, cursor, objs)
            events.append(ev)
            if hit is not None:
                self.tracker.mark_clicked(hit.id, now)           # clicked -> forgotten
                objs = [o for o in objs if o.id != hit.id]
                self.shared.log(f"{ev.kind} {ev.dt * 1000:+.0f}ms #{hit.id}")
            else:
                self.shared.log("stray click")

        # --- slider / spinner shaping signals
        follow_ok = None
        if ball is not None:
            follow_ok = bool(any(self.keys) and math.hypot(cursor[0] - ball.x, cursor[1] - ball.y) <= 2.4 * self.circle_r)
        spin_revs, spin_id = 0.0, None
        sp = next((o for o in objs if o.kind == SPINNER and o.ttl <= 0), None)
        if sp is not None:
            ang = math.atan2(cursor[1] - sp.y, cursor[0] - sp.x)
            if self._spin_ang is not None and math.hypot(cursor[0] - sp.x, cursor[1] - sp.y) > 20:
                spin_revs = abs((ang - self._spin_ang + math.pi) % (2 * math.pi) - math.pi) / (2 * math.pi)
            self._spin_ang, spin_id = ang, sp.id
        else:
            self._spin_ang = None
        r = self.shaper.step(events, cursor, objs, self.keys, follow_ok, spin_revs, spin_id, SPINNER_REV_CAP)

        if self.learner is not None:
            self.buf.add(obs, act, logp, val, r)
        self._n_steps += 1
        if self._n_steps % 60 == 0:
            c = self.judge.counts
            with self.shared.lock:
                self.shared.stats["judge"] = (f"{c['300']}/{c['100']}/{c['50']}/{c['miss']} stray {c['stray']} "
                                              f"acc {self.judge.accuracy() * 100:.1f}%")
