"""What the policy sees. The SAME encoder is used by the simulator, the replay dataset builder
and the live bot, so a policy trained on one works on the others.

The policy never sees a beatmap - only what a player sees on screen:
  * the cursor (position, velocity), which action keys are down and for how long
  * up to N upcoming objects: position relative to the cursor, time-to-hit, radius, kind
  * the slider ball, if one is on screen
All positions are osu! playfield pixels (512 x 384).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np

CIRCLE, SLIDER_HEAD, SPINNER = 0, 1, 2

G = 14      # global features
F = 11      # features per object

# global feature indices
I_CX, I_CY, I_VX, I_VY, I_K0, I_K1, I_A0, I_A1 = range(8)
I_BV, I_BDX, I_BDY, I_BVX, I_BVY, I_PRE = range(8, 14)
# per-object offsets
O_VALID, O_DX, O_DY, O_AX, O_AY, O_TTLC, O_TTLF, O_R, O_CIRC, O_SLID, O_SPIN = range(11)

VEL_SCALE = 1500.0   # px/s per unit


@dataclass
class VisObj:
    id: int
    kind: int          # CIRCLE | SLIDER_HEAD | SPINNER
    x: float
    y: float
    r: float
    ttl: float         # seconds until it must be hit (negative = overdue)


@dataclass
class VisBall:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0


def obs_dim(n_objs: int) -> int:
    return G + F * n_objs


def x_indices(n_objs: int):
    """indices whose sign flips under a horizontal mirror (and the y equivalents)."""
    xs = [I_CX, I_VX, I_BDX, I_BVX] + [G + F * i + o for i in range(n_objs) for o in (O_DX, O_AX)]
    ys = [I_CY, I_VY, I_BDY, I_BVY] + [G + F * i + o for i in range(n_objs) for o in (O_DY, O_AY)]
    return xs, ys


def encode(n_objs: int, cursor, vel, keys, key_age, objs: Sequence[VisObj],
           ball: Optional[VisBall], preempt: float, out: Optional[np.ndarray] = None) -> np.ndarray:
    o = np.zeros(obs_dim(n_objs), np.float32) if out is None else out
    o[:] = 0.0
    cx, cy = cursor
    o[I_CX], o[I_CY] = cx / 256.0 - 1.0, cy / 192.0 - 1.0
    o[I_VX], o[I_VY] = vel[0] / VEL_SCALE, vel[1] / VEL_SCALE
    o[I_K0], o[I_K1] = float(keys[0]), float(keys[1])
    o[I_A0], o[I_A1] = min(key_age[0], 0.3) / 0.3, min(key_age[1], 0.3) / 0.3
    if ball is not None:
        o[I_BV] = 1.0
        o[I_BDX], o[I_BDY] = (ball.x - cx) / 128.0, (ball.y - cy) / 128.0
        o[I_BVX], o[I_BVY] = ball.vx / VEL_SCALE, ball.vy / VEL_SCALE
    o[I_PRE] = preempt / 1.2
    for i, ob in enumerate(objs[:n_objs]):
        b = G + F * i
        ttl = ob.ttl
        o[b + O_VALID] = 1.0
        o[b + O_DX] = max(-4.0, min(4.0, (ob.x - cx) / 128.0))
        o[b + O_DY] = max(-4.0, min(4.0, (ob.y - cy) / 128.0))
        o[b + O_AX], o[b + O_AY] = ob.x / 256.0 - 1.0, ob.y / 192.0 - 1.0
        o[b + O_TTLC] = 2.0 * max(-0.4, min(1.6, ttl))
        o[b + O_TTLF] = math.tanh(ttl / 0.04)
        o[b + O_R] = ob.r / 64.0
        o[b + O_CIRC + ob.kind] = 1.0
    return o


def sort_visible(objs: list) -> list:
    """Soonest first (the order the policy expects)."""
    return sorted(objs, key=lambda v: v.ttl)
