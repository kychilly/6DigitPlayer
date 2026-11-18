"""Human-motor-control models: curved min-jerk trajectories, tremor, aim scatter,
and a mixture-of-normals tap timing model."""
from __future__ import annotations
import math
import numpy as np
from config import HumanConfig


class TremorNoise:
    """Deterministic-in-time noise: 8-12 Hz tremor + slow drift (callable at any rate)."""

    def __init__(self, rng: np.random.Generator, amp_px: float, drift_px: float):
        self.f = rng.uniform(7.5, 12.5, size=(2, 3))
        self.ph = rng.uniform(0, 2 * math.pi, size=(2, 3))
        self.a = amp_px * np.array([1.0, 0.6, 0.35])
        self.lf = rng.uniform(0.2, 1.8, size=(2, 3))
        self.lph = rng.uniform(0, 2 * math.pi, size=(2, 3))
        self.drift = drift_px / 3.0
        self.env_ph = rng.uniform(0, 2 * math.pi)

    def __call__(self, t: float) -> np.ndarray:
        env = 0.75 + 0.25 * math.sin(2 * math.pi * 0.4 * t + self.env_ph)
        out = np.zeros(2)
        for ax in range(2):
            fast = float(np.sum(self.a * np.sin(2 * math.pi * self.f[ax] * t + self.ph[ax])))
            slow = float(np.sum(np.sin(2 * math.pi * self.lf[ax] * t + self.lph[ax]))) * self.drift
            out[ax] = env * fast + slow
        return out


class Trajectory:
    """Quintic (minimum-jerk family) path from p0 to p1 with boundary velocities,
    plus a smooth sideways bulge so the path is an arc, not a line."""

    def __init__(self, p0, v0, p1, t0, T, rng, curve_frac, v1=None):
        self.p0 = np.asarray(p0, float)
        self.p1 = np.asarray(p1, float)
        self.v0 = np.asarray(v0, float)
        v1 = np.zeros(2) if v1 is None else np.asarray(v1, float)
        self.t0 = t0
        self.T = T = max(float(T), 1e-3)
        d = self.p1 - self.p0
        self.c3 = (20 * d - (8 * v1 + 12 * self.v0) * T) / (2 * T ** 3)
        self.c4 = (-30 * d + (14 * v1 + 16 * self.v0) * T) / (2 * T ** 4)
        self.c5 = (12 * d - 6 * (v1 + self.v0) * T) / (2 * T ** 5)
        L = float(np.linalg.norm(d))
        self.normal = np.array([-d[1], d[0]]) / L if L > 1e-6 else np.zeros(2)
        self.amp = curve_frac * L * (1 if rng.random() < 0.5 else -1)

    def position(self, t: float) -> np.ndarray:
        tau = min(max(t - self.t0, 0.0), self.T)
        p = self.p0 + self.v0 * tau + self.c3 * tau ** 3 + self.c4 * tau ** 4 + self.c5 * tau ** 5
        u = tau / self.T
        s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
        return p + self.normal * self.amp * math.sin(math.pi * s)

    def velocity(self, t: float) -> np.ndarray:
        h = 5e-4
        return (self.position(t + h) - self.position(t - h)) / (2 * h)

    def finished(self, t: float) -> bool:
        return t >= self.t0 + self.T

    def sample_path(self, n: int = 24) -> list:
        return [self.position(self.t0 + self.T * i / (n - 1)) for i in range(n)]


def sample_aim_point(center, radius, dist, T, rng, h: HumanConfig):
    """Where the cursor *actually* lands. Scatter grows with speed (signal-dependent
    noise); occasionally the aim is a genuine miss. Returns (point, is_miss)."""
    center = np.asarray(center, float)
    v_r = dist / max(T, 0.02) / max(radius, 1.0)  # radii per second
    sigma = radius * (h.aim_sigma_frac + h.aim_speed_gain * min(v_r / 10.0, 3.0))
    p_miss = min(0.4, h.miss_base + h.miss_gain * max(0.0, v_r - 12.0) / 10.0)
    if rng.random() < p_miss:
        ang = rng.uniform(0, 2 * math.pi)
        off = radius * rng.uniform(1.1, 1.7) * np.array([math.cos(ang), math.sin(ang)])
        return center + off, True
    off = rng.normal(0.0, sigma, size=2)
    return center + off, float(np.linalg.norm(off)) > 0.95 * radius


class TapTimingModel:
    """Tap offset relative to the true hit time. Mixture: tight normal for 300s,
    wider uniform bands for 100s / 50s, with OD-dependent hit windows."""

    def __init__(self, od: float, h: HumanConfig, rng: np.random.Generator):
        self.h, self.rng = h, rng
        self.w300 = (80 - 6 * od) / 1000.0
        self.w100 = (140 - 8 * od) / 1000.0
        self.w50 = (200 - 10 * od) / 1000.0

    def sample(self, stream: bool = False) -> float:
        h, rng = self.h, self.rng
        p100 = h.p100 * (h.stream_p100_mult if stream else 1.0)
        r = rng.random()
        if r < h.p50:
            mag = rng.uniform(self.w100, self.w50 * 0.97)
            return mag * (1 if rng.random() < 0.6 else -1)  # slightly late-biased
        if r < h.p50 + p100:
            mag = rng.uniform(self.w300, self.w100 * 0.97)
            return mag * (1 if rng.random() < 0.6 else -1)
        lim = self.w300 * 0.97
        return float(np.clip(rng.normal(h.tap_mu, self.w300 / 2.8), -lim, lim))

    def hold_time(self) -> float:
        return max(0.02, self.rng.normal(self.h.hold_mean, self.h.hold_std))
