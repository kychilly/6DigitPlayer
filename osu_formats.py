"""Parsers for osu! beatmaps (.osu) and replays (.osr).

These are used ONLY during training / simulation. At play time the bot never reads a map: it
sees objects through the screen-capture detector, exactly like a human.

Coordinates are osu! playfield pixels (512 x 384).
"""
from __future__ import annotations
import lzma
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

CIRCLE, SLIDER, SPINNER = 0, 1, 2          # hit-object kinds (not the detector classes)
PLAYFIELD_W, PLAYFIELD_H = 512.0, 384.0

# mod bit flags
MOD_NF, MOD_EZ, MOD_TD, MOD_HD, MOD_HR, MOD_SD = 1, 2, 4, 8, 16, 32
MOD_DT, MOD_RX, MOD_HT, MOD_NC, MOD_FL, MOD_AT = 64, 128, 256, 512, 1024, 2048
MOD_SO, MOD_AP, MOD_PF = 4096, 8192, 16384
BAD_MODS = MOD_DT | MOD_RX | MOD_HT | MOD_NC | MOD_AT | MOD_AP   # skipped when learning from replays


# ====================================================================== beatmap
@dataclass
class HitObject:
    kind: int
    x: float
    y: float
    time: float                 # seconds
    end_time: float             # seconds (== time for circles)
    slides: int = 1
    path: Optional[np.ndarray] = None       # (N,2) dense polyline, arc-length resampled
    # filled by Beatmap.finalize
    index: int = 0

    def ball_pos(self, t: float) -> np.ndarray:
        """Slider ball position at absolute time t (seconds)."""
        if self.path is None or self.end_time <= self.time:
            return np.array([self.x, self.y])
        p = (t - self.time) / (self.end_time - self.time)
        p = min(max(p, 0.0), 1.0) * self.slides
        k = min(int(p), self.slides - 1)
        u = p - k
        if k % 2 == 1:
            u = 1.0 - u
        f = u * (len(self.path) - 1)
        i = min(int(f), len(self.path) - 2)
        a = f - i
        return self.path[i] * (1 - a) + self.path[i + 1] * a


@dataclass
class Beatmap:
    title: str = ""
    cs: float = 4.0
    ar: float = 9.0
    od: float = 8.0
    hp: float = 5.0
    slider_multiplier: float = 1.4
    slider_tick_rate: float = 1.0
    objects: list = field(default_factory=list)
    path: str = ""

    def with_mods(self, mods: int) -> "Beatmap":
        """Return a copy with HR / EZ difficulty changes and y-flip applied."""
        import copy
        bm = copy.deepcopy(self)
        if mods & MOD_HR:
            bm.cs, bm.ar, bm.od = min(10, bm.cs * 1.3), min(10, bm.ar * 1.4), min(10, bm.od * 1.4)
            for o in bm.objects:
                o.y = PLAYFIELD_H - o.y
                if o.path is not None:
                    o.path = o.path.copy()
                    o.path[:, 1] = PLAYFIELD_H - o.path[:, 1]
        elif mods & MOD_EZ:
            bm.cs, bm.ar, bm.od = bm.cs * 0.5, bm.ar * 0.5, bm.od * 0.5
        return bm

    @property
    def radius(self) -> float:
        return 54.4 - 4.48 * self.cs


def _bezier_point(pts: np.ndarray, t: float) -> np.ndarray:
    p = pts.copy()
    for k in range(len(p) - 1, 0, -1):
        p[:k] = p[:k] * (1 - t) + p[1:k + 1] * t
    return p[0]


def _bezier_poly(pts: np.ndarray, n: int | None = None) -> np.ndarray:
    n = n or max(16, min(120, len(pts) * 20))
    return np.array([_bezier_point(pts, i / (n - 1)) for i in range(n)])


def _perfect_circle(p: np.ndarray, n: int = 80) -> Optional[np.ndarray]:
    (ax, ay), (bx, by), (cx, cy) = p
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-6:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    r = math.hypot(ax - ux, ay - uy)
    a0, a1, a2 = (math.atan2(q[1] - uy, q[0] - ux) for q in p)
    two_pi = 2 * math.pi
    # choose direction so the arc passes through the middle point
    if not (((a1 - a0) % two_pi) < ((a2 - a0) % two_pi)):
        span = -((a0 - a2) % two_pi)
    else:
        span = (a2 - a0) % two_pi
    ts = np.linspace(0, 1, n)
    ang = a0 + span * ts
    return np.stack([ux + r * np.cos(ang), uy + r * np.sin(ang)], 1)


def _catmull_poly(pts: np.ndarray, n_per: int = 16) -> np.ndarray:
    out = []
    for i in range(len(pts) - 1):
        p0 = pts[max(i - 1, 0)]; p1 = pts[i]; p2 = pts[i + 1]; p3 = pts[min(i + 2, len(pts) - 1)]
        for k in range(n_per):
            t = k / n_per
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(pts[-1])
    return np.array(out)


def _resample(poly: np.ndarray, length: float, n: int = 64) -> np.ndarray:
    """Arc-length resample to exactly `length` osu!px (truncate or extend along the last tangent)."""
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-6:
        return np.repeat(poly[:1], 2, axis=0)
    targets = np.linspace(0, length, n)
    out = np.empty((n, 2))
    for j, s in enumerate(targets):
        if s <= total:
            i = min(np.searchsorted(cum, s, side="right") - 1, len(seg) - 1)
            a = (s - cum[i]) / max(seg[i], 1e-9)
            out[j] = poly[i] * (1 - a) + poly[i + 1] * a
        else:  # extend along the final direction
            d = poly[-1] - poly[-2]
            d = d / max(np.linalg.norm(d), 1e-9)
            out[j] = poly[-1] + d * (s - total)
    return out


def _slider_path(curve: str, pts: np.ndarray, length: float) -> np.ndarray:
    if curve == "L":
        poly = pts
    elif curve == "P" and len(pts) == 3:
        poly = _perfect_circle(pts)
        if poly is None:
            poly = pts
    elif curve == "C":
        poly = _catmull_poly(pts)
    else:  # bezier; red anchors (repeated points) split segments
        segs, cur = [], [pts[0]]
        for i in range(1, len(pts)):
            cur.append(pts[i])
            if i < len(pts) - 1 and np.allclose(pts[i], pts[i + 1]):
                segs.append(np.array(cur)); cur = []
        if cur:
            segs.append(np.array(cur))
        poly = np.concatenate([_bezier_poly(s) if len(s) > 2 else s for s in segs if len(s) > 1], axis=0)
    return _resample(np.asarray(poly, float), length)


def parse_osu(path: str | Path) -> Beatmap:
    bm = Beatmap(path=str(path))
    section = ""
    timing = []   # (time_ms, beat_length, uninherited)
    raw_objects = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            if line.startswith("["):
                section = line.strip("[]")
                continue
            if section == "Metadata" and line.startswith("Title:"):
                bm.title = line.split(":", 1)[1].strip()
            elif section == "Difficulty" and ":" in line:
                k, v = (s.strip() for s in line.split(":", 1))
                v = float(v)
                if k == "CircleSize": bm.cs = v
                elif k == "ApproachRate": bm.ar = v
                elif k == "OverallDifficulty": bm.od = v
                elif k == "HPDrainRate": bm.hp = v
                elif k == "SliderMultiplier": bm.slider_multiplier = v
                elif k == "SliderTickRate": bm.slider_tick_rate = v
            elif section == "TimingPoints":
                p = line.split(",")
                if len(p) >= 2:
                    unin = (int(p[6]) == 1) if len(p) > 6 else (float(p[1]) > 0)
                    timing.append((float(p[0]), float(p[1]), unin))
            elif section == "HitObjects":
                raw_objects.append(line.split(","))
    if "ApproachRate" not in open(path, "r", encoding="utf-8-sig", errors="replace").read():
        bm.ar = bm.od   # very old maps: AR == OD
    timing.sort(key=lambda t: t[0])

    def beat_and_sv(t_ms: float):
        beat, sv = 500.0, 1.0
        for tm, bl, unin in timing:
            if tm > t_ms + 1e-6:
                break
            if unin:
                beat, sv = bl, 1.0
            else:
                sv = 100.0 / min(max(-bl, 10.0), 1000.0)
        return beat, sv

    for p in raw_objects:
        try:
            x, y, t_ms, typ = float(p[0]), float(p[1]), float(p[2]), int(p[3])
        except (ValueError, IndexError):
            continue
        if typ & 8:  # spinner
            bm.objects.append(HitObject(SPINNER, 256.0, 192.0, t_ms / 1000, float(p[5]) / 1000))
        elif typ & 2:  # slider
            try:
                cp = p[5].split("|")
                pts = np.array([[x, y]] + [[float(a) for a in q.split(":")] for q in cp[1:]], float)
                slides, length = int(p[6]), float(p[7])
            except (ValueError, IndexError):
                continue
            beat, sv = beat_and_sv(t_ms)
            dur = length / (bm.slider_multiplier * 100.0 * sv) * beat * slides / 1000.0
            bm.objects.append(HitObject(SLIDER, x, y, t_ms / 1000, t_ms / 1000 + dur, slides,
                                        _slider_path(cp[0], pts, length)))
        else:
            bm.objects.append(HitObject(CIRCLE, x, y, t_ms / 1000, t_ms / 1000))
    bm.objects.sort(key=lambda o: o.time)
    for i, o in enumerate(bm.objects):
        o.index = i
    return bm


# ====================================================================== replay
@dataclass
class Replay:
    mode: int
    version: int
    beatmap_md5: str
    player: str
    mods: int
    n300: int
    n100: int
    n50: int
    nmiss: int
    t: np.ndarray               # seconds, monotonic
    x: np.ndarray
    y: np.ndarray
    keys: np.ndarray            # int bitmask
    path: str = ""

    def key_state(self, i: int) -> np.ndarray:
        """bool array for action key i (0: M1/K1, 1: M2/K2)."""
        m = 0b0101 if i == 0 else 0b1010
        return (self.keys & m) != 0


def _read_string(b: bytes, o: int):
    if b[o] == 0:
        return "", o + 1
    o += 1
    n, shift = 0, 0
    while True:                       # ULEB128
        c = b[o]; o += 1
        n |= (c & 0x7F) << shift
        shift += 7
        if not c & 0x80:
            break
    return b[o:o + n].decode("utf-8", "replace"), o + n


def parse_osr(path: str | Path) -> Replay:
    b = Path(path).read_bytes()
    o = 0
    mode = b[o]; o += 1
    version = struct.unpack_from("<i", b, o)[0]; o += 4
    md5, o = _read_string(b, o)
    player, o = _read_string(b, o)
    _, o = _read_string(b, o)
    n300, n100, n50, _geki, _katu, nmiss = struct.unpack_from("<6h", b, o); o += 12
    o += 4 + 2 + 1                                  # score, max combo, perfect
    mods = struct.unpack_from("<i", b, o)[0]; o += 4
    _, o = _read_string(b, o)                       # life graph
    o += 8                                          # timestamp
    n = struct.unpack_from("<i", b, o)[0]; o += 4
    text = lzma.decompress(b[o:o + n]).decode("ascii", "ignore")
    ts, xs, ys, zs, cur = [], [], [], [], 0.0
    for fr in text.split(","):
        p = fr.split("|")
        if len(p) < 4:
            continue
        try:
            w, x, y, z = int(p[0]), float(p[1]), float(p[2]), int(float(p[3]))
        except ValueError:
            continue
        if w == -12345:                              # RNG seed frame
            continue
        cur += w
        if y <= -400 and not ts:                     # leading dummy frames
            continue
        ts.append(cur); xs.append(x); ys.append(y); zs.append(z)
    t = np.maximum.accumulate(np.array(ts, float) / 1000.0)
    return Replay(mode, version, md5, player, mods, n300, n100, n50, nmiss,
                  t, np.array(xs), np.array(ys), np.array(zs, int), str(path))
