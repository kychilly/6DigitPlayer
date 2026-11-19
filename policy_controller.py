"""Plays real osu! with the LEARNED policy (replaces the hand-written humanizer controller).

Every 1/policy.hz seconds it builds the same observation the simulator uses - from the screen
tracker only, never a map - asks the network for [aim velocity, key0, key1], and applies it.
At 1 kHz the cursor follows that velocity with a hand-acceleration limit plus physiological
tremor, so the motion stays smooth and human.

It also JUDGES its own taps the way osu! would (300/100/50/miss from the tracker's timing + cursor
position) for the on-screen stats. With a learner attached it records every decision (observation, action,
log-prob, value, time, key presses); at the end of the map the episode is handed to the learner
(EpisodeRecorder -> rl_worker.py), which re-scores it with osu!'s real judgement from the replay and runs PPO.
"""
from __future__ import annotations
import math
import traceback

import numpy as np

from config import BotConfig, ar_to_preempt
from controller import Controller, Mode
from judge import LiveJudge, RewardShaper, Event, hit_windows
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
        self._diff_ver = -1
        self.map_name = ""
        self._last_now = 0.0
        self._paused_at = None        # map paused (break section / no notes) - the episode stays open for a while
        self._paused_ver = -1
        self._map_end_seen = getattr(shared, "map_end_ver", 0)
        if hasattr(shared, "on_kill"):
            shared.on_kill.append(self.emergency_save)

    # ------------------------------------------------------------------ difficulty (AR / OD) for the current map
    def set_difficulty(self, ar: float, od: float, name: str = ""):
        """Called when the map's AR / OD are known (auto-detected from the osu! window title, or --ar/--od)."""
        self.cfg.game.ar, self.cfg.game.od = ar, od
        self.preempt = ar_to_preempt(ar)
        self.judge.od = od
        self.judge.w300, self.judge.w100, self.judge.w50 = hit_windows(od)
        self.shaper.od = od
        self.shaper.w300, self.shaper.w100, self.shaper.w50 = hit_windows(od)
        if hasattr(self.tracker, "set_preempt"):
            self.tracker.set_preempt(self.preempt)
        self.map_name = name

    def _sync_difficulty(self):
        d = getattr(self.shared, "diff", None)
        ver = getattr(self.shared, "diff_ver", 0)
        if d is not None and ver != self._diff_ver:
            self._diff_ver = ver
            ar, od, name = d
            self.set_difficulty(ar, od, name)
            self.shared.log(f"map: {name[:40]} | AR {ar:.1f} OD {od:.1f}")

    # ------------------------------------------------------------------ main step (1 kHz)
    def step(self, now: float, dt: float):
        armed = self.shared.armed
        self._last_now = now
        self._sync_difficulty()
        mev = getattr(self.shared, "map_end_ver", 0)
        if mev != self._map_end_seen:                  # results screen / left the map: the play is over NOW
            self._map_end_seen = mev
            if self.active or self._paused_at is not None:
                if self.active:
                    self._disengage(getattr(self.shared, "map_end_reason", "") or "map over")
                self.flush_episode()
        if self._paused_at is not None and (now - self._paused_at > self.pc.episode_gap
                                            or self._paused_ver != self._diff_ver):
            self.flush_episode()                       # the map is really over (or a new one started)
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
        if self._paused_at is not None:                # back after a break in the SAME map: keep recording
            self._paused_at = None
            return
        self.judge.reset(); self.shaper.reset()
        self.buf = ExperienceBuffer()
        self._n_steps = 0

    def _end_episode(self):
        """Control released (no notes / focus lost / disarmed). The map may just be in a break, so the episode
        is only paused; flush_episode() closes it when the map is really over."""
        if not self.active:
            return
        self.active = False
        self._paused_at, self._paused_ver = self._last_now, self._diff_ver

    def flush_episode(self, sync: bool = False):
        """The map is over: report it and hand the recording to the learner."""
        self._paused_at = None
        if not len(self.buf) and not any(self.judge.counts.values()):
            return
        c = dict(self.judge.counts)
        acc = self.judge.accuracy()
        self.shared.log(f"map finished: acc {acc * 100:.1f}% {c['300']}/{c['100']}/{c['50']}/{c['miss']} (300/100/50/miss)")
        if self.learner is not None:
            try:
                self.learner.finish_episode(self.buf, acc, c, {"ar": self.cfg.game.ar, "od": self.cfg.game.od,
                                                               "map": self.map_name}, sync=sync)
            except Exception:
                traceback.print_exc()
        self.buf = ExperienceBuffer()
        self.judge.reset(); self.shaper.reset()
        self._n_steps = 0

    def emergency_save(self):
        """Kill switch / shutdown: save the play in progress (synchronously - the process is about to exit)."""
        if self.active:
            self.active = False
            self._paused_at = self._last_now
        if self._paused_at is not None:
            self.flush_episode(sync=True)

    def stop(self):
        try:
            self.emergency_save()                      # save the last map on a normal shutdown
        except Exception:
            traceback.print_exc()
        super().stop()

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
            self.buf.add(obs, act, logp, val, r, now, (0 in rising, 1 in rising))
        self._n_steps += 1
        if self._n_steps % 60 == 0:
            c = self.judge.counts
            with self.shared.lock:
                self.shared.stats["judge"] = (f"{c['300']}/{c['100']}/{c['50']}/{c['miss']} stray {c['stray']} "
                                              f"acc {self.judge.accuracy() * 100:.1f}%")
