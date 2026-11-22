"""Random osu!-like beatmaps (circles, jumps, streams, sliders, spinners) so the simulator
never runs out of training maps. Real .osu files can be mixed in with --maps."""
from __future__ import annotations
import copy
import math
from pathlib import Path

import numpy as np

from osu_formats import (Beatmap, HitObject, CIRCLE, SLIDER, SPINNER, parse_osu, _slider_path)

MARGIN = 30.0


def _clamp(p):
    return np.array([min(max(p[0], MARGIN), 512 - MARGIN), min(max(p[1], MARGIN), 384 - MARGIN)])


def random_beatmap(rng: np.random.Generator, seconds: float = 40.0, difficulty: float | None = None) -> Beatmap:
    """difficulty in [0,1] scales speed / spacing / stream length."""
    d = rng.uniform(0.15, 1.0) if difficulty is None else difficulty
    bm = Beatmap(title="generated", cs=float(rng.uniform(3.0, 6.0)), ar=float(rng.uniform(7.5, 10.0)),
                 od=float(rng.uniform(5.0, 9.5)), slider_multiplier=1.6)
    bpm = 100 + 120 * d * rng.uniform(0.7, 1.0)
    beat = 60.0 / bpm
    t = 1.5
    pos = np.array([256.0, 192.0]) + rng.normal(0, 60, 2)
    heading = rng.uniform(0, 2 * math.pi)
    objs: list[HitObject] = []

    def add_circle(p, tt):
        objs.append(HitObject(CIRCLE, float(p[0]), float(p[1]), tt, tt))

    while t < seconds:
        w = np.array([0.25, 0.22, 0.12 + 0.12 * d, 0.28 - 0.06 * d, 0.03, 0.10])
        pat = rng.choice(["single", "jump", "stream", "slider", "spinner", "triple"], p=w / w.sum())
        if pat == "single":
            step = rng.uniform(60, 220)
            heading += rng.normal(0, 1.0)
            pos = _clamp(pos + step * np.array([math.cos(heading), math.sin(heading)]))
            add_circle(pos, t)
            t += beat * rng.choice([1.0, 0.5, 1.0])
        elif pat == "jump":
            for _ in range(int(rng.integers(2, 6))):
                step = rng.uniform(120, 160 + 220 * d)
                heading = rng.uniform(0, 2 * math.pi)
                pos = _clamp(pos + step * np.array([math.cos(heading), math.sin(heading)]))
                add_circle(pos, t)
                t += beat * rng.choice([0.5, 0.5, 1.0])
        elif pat == "stream":
            n = int(rng.integers(4, 6 + int(14 * d)))
            sp = rng.uniform(18, 70)
            for _ in range(n):
                heading += rng.normal(0, 0.15)
                pos = _clamp(pos + sp * np.array([math.cos(heading), math.sin(heading)]))
                add_circle(pos, t)
                t += beat * 0.25
            t += beat * 0.25
        elif pat == "triple":
            for _ in range(3):
                heading += rng.normal(0, 0.6)
                pos = _clamp(pos + rng.uniform(40, 120) * np.array([math.cos(heading), math.sin(heading)]))
                add_circle(pos, t)
                t += beat * 0.5
        elif pat == "slider":
            length = rng.uniform(70, 280)
            curve = rng.choice(["L", "P", "B"], p=[0.4, 0.3, 0.3])
            ang = rng.uniform(0, 2 * math.pi)
            p0 = _clamp(pos)
            dirv = np.array([math.cos(ang), math.sin(ang)])
            if curve == "L":
                pts = np.array([p0, p0 + dirv * length])
            elif curve == "P":
                nrm = np.array([-dirv[1], dirv[0]]) * rng.uniform(-0.5, 0.5) * length
                pts = np.array([p0, p0 + dirv * length * 0.5 + nrm, p0 + dirv * length])
            else:
                nrm = np.array([-dirv[1], dirv[0]])
                pts = np.array([p0, p0 + dirv * length * 0.33 + nrm * rng.uniform(-60, 60),
                                p0 + dirv * length * 0.66 + nrm * rng.uniform(-60, 60), p0 + dirv * length])
            pts = np.array([_clamp(p) for p in pts])
            path = _slider_path(curve, pts, length)
            path = np.array([_clamp(p) for p in path])
            slides = int(rng.choice([1, 1, 1, 2, 3]))
            dur = length / (bm.slider_multiplier * 100.0) * beat * slides
            objs.append(HitObject(SLIDER, float(path[0][0]), float(path[0][1]), t, t + dur, slides, path,
                                  tick_interval=beat))
            pos = path[-1] if slides % 2 == 1 else path[0]
            t += dur + beat * rng.choice([0.5, 1.0])
        else:  # spinner
            dur = rng.uniform(1.2, 2.8)
            objs.append(HitObject(SPINNER, 256.0, 192.0, t, t + dur))
            pos = np.array([256.0, 192.0]) + rng.normal(0, 40, 2)
            t += dur + beat
    for i, o in enumerate(objs):
        o.index = i
    bm.objects = objs
    return bm


def segment(bm: Beatmap, seconds: float, rng: np.random.Generator) -> Beatmap:
    """A random window of a real map (at most `seconds` long)."""
    objs = bm.objects
    if not objs or objs[-1].end_time - objs[0].time <= seconds:
        return bm
    start_t = rng.uniform(objs[0].time, objs[-1].end_time - seconds)
    sel = [o for o in objs if start_t <= o.time <= start_t + seconds]
    if not sel:
        return bm
    out = copy.copy(bm)
    out.objects = [copy.copy(o) for o in sel]
    for i, o in enumerate(out.objects):
        o.index = i
    return out


def load_maps(folder: str | None, limit: int = 400) -> list[Beatmap]:
    if not folder:
        return []
    maps = []
    for p in list(Path(folder).rglob("*.osu"))[:limit]:
        try:
            bm = parse_osu(p)
            if len(bm.objects) >= 20:
                maps.append(bm)
        except Exception as e:  # noqa
            print(f"[maps] skipped {p.name}: {e}")
    print(f"[maps] loaded {len(maps)} beatmaps from {folder}")
    return maps
