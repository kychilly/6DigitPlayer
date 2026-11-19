"""Training frames drawn with YOUR osu! skin's real sprites, so the detector learns what your circles actually look like.

    python synthetic_preview.py --skin "C:/Users/you/AppData/Local/osu!/Skins/<skin>"   # eyeball it first
    python train.py --skin "C:/Users/you/AppData/Local/osu!/Skins/<skin>" --resume checkpoints/detector.pt \
                    --epochs 15 --out checkpoints/detector_skin.pt

Follows osu!stable's skinning rules: hitcircle(.png) tinted with the combo colour + hitcircleoverlay on top +
the combo number (HitCirclePrefix from skin.ini); approachcircle tinted and shrinking from 4x to 1x; fade-in;
slider bodies in the skin's SliderBorder / SliderTrackOverride colours, sliderendcircle as a hard negative;
sliderb + sliderfollowcircle; spinner-background / spinner-circle; the skin's cursor; and the skin's
hit100/hit50/hit0 popups as distractors. A 128 px (@1x) circle sprite = one circle diameter, like in osu!.
Labels are identical to synthetic.SyntheticRenderer, so train.py doesn't change.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import cv2
import numpy as np

from synthetic import GTObject, SyntheticRenderer


def imread_rgba(path: Path):
    """cv2.imread can't open non-ASCII paths on Windows (your skin folder has 『』 in it) -> read bytes ourselves."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    if img.shape[0] <= 2 and img.shape[1] <= 2:     # skins use 1x1 transparent PNGs to hide an element
        return None
    return img


class Skin:
    """Loads the sprites + skin.ini of one skin folder. Missing / blank sprites become None."""

    def __init__(self, folder: str):
        self.dir = Path(folder)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"skin folder not found: {folder}")
        self.ini = self._read_ini()
        self.cache = {}
        g = lambda name: self.sprite(name)
        self.hitcircle = g("hitcircle")
        self.overlay = g("hitcircleoverlay")
        self.slider_start = g("sliderstartcircle") or self.hitcircle
        self.slider_start_overlay = g("sliderstartcircleoverlay") or self.overlay
        self.slider_end = g("sliderendcircle")
        self.approach = g("approachcircle")
        self.sliderb = g("sliderb") or g("sliderb0")
        self.follow = g("sliderfollowcircle") or g("sliderfollowcircle-0")
        self.reverse = g("reversearrow")
        self.spin_bg = g("spinner-background")
        self.spin_circle = g("spinner-circle")
        self.cursor = g("cursor")
        self.popups = [s for s in (g("hit100-0") or g("hit100"), g("hit50-0") or g("hit50"), g("hit0-0") or g("hit0"))
                       if s is not None]
        prefix = self.ini.get("Fonts", {}).get("HitCirclePrefix", "default").replace("\\", "/")
        self.digits = [self.sprite(f"{prefix}-{d}") for d in range(10)]
        if not any(x is not None for x in self.digits):                  # prefix wrong: look anywhere in the skin
            hits = sorted(self.dir.rglob("default-1*.png"))
            if hits:
                base = hits[0].parent / "default"
                self.digits = [self._load_variants(Path(f"{base}-{d}")) for d in range(10)]
        self.overlap = float(self.ini.get("Fonts", {}).get("HitCircleOverlap", 2) or 2)
        gen = self.ini.get("General", {})
        self.overlay_above_number = str(gen.get("HitCircleOverlayAboveNumber",
                                                gen.get("HitCircleOverlayAboveNumer", "1"))).strip() != "0"
        col = self.ini.get("Colours", {})
        self.combo_colours = [c for c in (self._col(col.get(f"Combo{i}")) for i in range(1, 9)) if c]
        self.slider_border = self._col(col.get("SliderBorder")) or (255, 255, 255)
        self.slider_track = self._col(col.get("SliderTrackOverride"))
        self.slider_ball_col = self._col(col.get("SliderBall"))
        if self.hitcircle is None:
            raise RuntimeError(f"no hitcircle.png in {folder} - is this an osu! skin folder?")

    # ---------------------------------------------------------------- loading
    def _read_ini(self) -> dict:
        p = self.dir / "skin.ini"
        out, sec = {}, ""
        if not p.exists():
            return out
        try:
            raw = np.fromfile(str(p), dtype=np.uint8).tobytes().decode("utf-8-sig", "replace")
        except OSError:
            return out
        for line in raw.splitlines():
            line = line.split("//")[0].strip()
            if not line:
                continue
            if line.startswith("["):
                sec = line.strip("[]")
                continue
            if ":" in line:
                k, v = (s.strip() for s in line.split(":", 1))
                out.setdefault(sec, {})[k] = v
        return out

    @staticmethod
    def _col(v):
        if not v:
            return None
        nums = [int(x) for x in re.findall(r"\d+", v)[:3]]
        return tuple(nums) if len(nums) == 3 else None          # RGB

    def _load_variants(self, stem: Path):
        """Prefer @2x (returns sprite at @1x-equivalent scale info)."""
        for suffix, scale in (("@2x.png", 2.0), (".png", 1.0)):
            img = imread_rgba(Path(str(stem) + suffix))
            if img is not None:
                return img, scale
        return None

    def sprite(self, name: str):
        if name not in self.cache:
            self.cache[name] = self._load_variants(self.dir / name)
        return self.cache[name]


# ====================================================================== drawing helpers
def paste(img, sprite, cx, cy, scale, tint=None, alpha=1.0, angle=0.0):
    """Alpha-composite `sprite` ((rgba, density) tuple) centred at (cx, cy).
    scale = output pixels per @1x sprite pixel (osu!: a hit circle is drawn at diameter/128)."""
    if sprite is None or alpha <= 0.01:
        return
    rgba, dens = sprite
    s = scale / dens                                     # output pixels per actual sprite pixel
    w, h = max(1, int(round(rgba.shape[1] * s))), max(1, int(round(rgba.shape[0] * s)))
    if w > 4 * img.shape[1] or h > 4 * img.shape[0]:
        return
    spr = cv2.resize(rgba, (w, h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    if angle:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        spr = cv2.warpAffine(spr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    H, W = img.shape[:2]
    ax0, ay0, ax1, ay1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if ax0 >= ax1 or ay0 >= ay1:
        return
    sub = spr[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0].astype(np.float32)
    rgb = sub[:, :, :3]
    if tint is not None:                                 # osu! tints by multiplying (tint is RGB, image is BGR)
        rgb = rgb * (np.array(tint[::-1], np.float32) / 255.0)
    a = (sub[:, :, 3:4] / 255.0) * alpha
    dst = img[ay0:ay1, ax0:ax1].astype(np.float32)
    img[ay0:ay1, ax0:ax1] = np.clip(rgb * a + dst * (1 - a), 0, 255).astype(np.uint8)


class SkinRenderer(SyntheticRenderer):
    """Same scene logic and labels as SyntheticRenderer, drawn with a real skin."""

    def __init__(self, skin_dir: str, W=384, H=224, stride=4, num_classes=4):
        super().__init__(W, H, stride, num_classes)
        self.skin = Skin(skin_dir)

    def _combo_col(self, rng):
        sk = self.skin
        if sk.combo_colours and rng.random() < 0.7:
            return sk.combo_colours[int(rng.integers(len(sk.combo_colours)))]
        c = self._color(rng)                             # beatmaps can override combo colours
        return (c[2], c[1], c[0])

    def _number(self, img, x, y, n, d, alpha):
        sk = self.skin
        digs = [sk.digits[int(ch)] for ch in str(n)]
        if any(g is None for g in digs):
            return
        widths = [g[0].shape[1] / g[1] for g in digs]
        scale = d / 128.0 * 0.8                           # numbers are drawn at 0.8x circle scale
        total = sum(widths) * scale - sk.overlap * scale * (len(digs) - 1)
        cx = x - total / 2
        for g, wd in zip(digs, widths):
            paste(img, g, cx + wd * scale / 2, y, scale, alpha=alpha)
            cx += wd * scale - sk.overlap * scale

    def _circle(self, img, x, y, d, col, number, alpha, slider=False):
        sk = self.skin
        body = sk.slider_start if slider else sk.hitcircle
        over = sk.slider_start_overlay if slider else sk.overlay
        c = d / 128.0
        paste(img, body, x, y, c, tint=col, alpha=alpha)
        if not sk.overlay_above_number:
            paste(img, over, x, y, c, alpha=alpha)
        if number is not None:
            self._number(img, x, y, number, d, alpha)
        if sk.overlay_above_number:
            paste(img, over, x, y, c, alpha=alpha)

    def _body(self, img, pts, r, col):
        sk = self.skin
        track = sk.slider_track or tuple(int(v * 0.35) for v in col)
        border = sk.slider_border
        ip = pts.astype(np.int32).reshape(-1, 1, 2)
        for rad, c in ((r, border), (r * 0.86, track)):
            bgr = (int(c[2]), int(c[1]), int(c[0]))
            cv2.polylines(img, [ip], False, bgr, max(1, int(2 * rad)), cv2.LINE_AA)
            for p in ip[:, 0]:
                cv2.circle(img, (int(p[0]), int(p[1])), int(rad), bgr, -1, cv2.LINE_AA)

    # ---------------------------------------------------------------- render
    def render(self, rng):
        sk = self.skin
        W, H = self.W, self.H
        img = self._background(rng)
        objs: list[GTObject] = []
        r = float(rng.uniform(9, 20))                    # circle radius in model pixels (CS ~2..7 at 224p)
        d = 2 * r
        fade = float(rng.uniform(0.2, 0.6))              # fade-in as a fraction of the approach time
        taken = []

        def free_pos(rr, tries=12):
            for _ in range(tries):
                x, y = rng.uniform(W * 0.24, W * 0.76), rng.uniform(H * 0.18, H * 0.82)
                if all(math.hypot(x - a, y - b) > 2.3 * rr for a, b in taken):
                    return x, y
            return None

        def alpha_for(p):                                # p: 1 = just appeared, 0 = hit time
            return float(np.clip((1.0 - p) / fade, 0.0, 1.0))

        menu = rng.random() < 0.10
        spinner = (not menu) and rng.random() < 0.06
        draws = []
        if spinner:
            R = rng.uniform(0.28, 0.38) * H
            cx, cy = W / 2 + rng.normal(0, 3), H / 2 + rng.normal(0, 3)
            p = rng.uniform(0.0, 1.0)
            draws.append((p, "spinner", (cx, cy, R)))
            objs.append(GTObject(2, cx, cy, p, R))
        n_circ = 0 if (spinner or menu) else int(rng.integers(0, 6))
        n_slid = 0 if (spinner or menu) else int(rng.integers(0, 3))
        n_ball = 1 if (not (spinner or menu) and rng.random() < 0.35) else 0
        used_p, number = [], int(rng.integers(1, 6))
        for _ in range(n_circ):
            pos = free_pos(r)
            if pos is None:
                continue
            p = float(rng.uniform(0.02, 1.0))
            if any(abs(p - q) < 0.08 for q in used_p):
                continue
            used_p.append(p); taken.append(pos)
            draws.append((p, "circle", (pos[0], pos[1], self._combo_col(rng), number)))
            objs.append(GTObject(0, pos[0], pos[1], p, r))
            number += 1
        for i in range(n_slid + n_ball):
            pos = free_pos(r * 1.2)
            if pos is None:
                continue
            taken.append(pos)
            pts = self._slider_path(rng, pos[0], pos[1], r)
            for q in pts[::4]:
                taken.append((q[0], q[1]))
            col = self._combo_col(rng)
            if i < n_slid:
                p = float(rng.uniform(0.02, 1.0))
                draws.append((p, "slider", (pts, col, number)))
                objs.append(GTObject(1, pts[0, 0], pts[0, 1], p, r))
                number += 1
            else:
                k = int(rng.integers(1, len(pts) - 1))
                draws.append((0.0, "ball", (pts, k, col)))
                objs.append(GTObject(3, pts[k, 0], pts[k, 1], 0.0, r))

        self._clutter(img, rng, r)                       # generic HUD clutter (score, combo, hp bar)
        for _ in range(int(rng.integers(0, 3))):         # the skin's own 100 / 50 / miss popups
            if sk.popups:
                spr = sk.popups[int(rng.integers(len(sk.popups)))]
                paste(img, spr, rng.uniform(W * 0.2, W * 0.8), rng.uniform(H * 0.2, H * 0.8),
                      d / 128.0 * rng.uniform(0.8, 1.6), alpha=float(rng.uniform(0.4, 1.0)))
        if menu:
            self._logo(img, rng)

        # far-future objects are drawn underneath (osu! draws later notes below earlier ones)
        for p, kind, dd in sorted(draws, key=lambda t: -t[0]):
            a = alpha_for(p)
            if kind == "spinner":
                cx, cy, R = dd
                fit = lambda spr, size: size / (spr[0].shape[1] / spr[1]) if spr is not None else 0
                paste(img, sk.spin_bg, cx, cy, fit(sk.spin_bg, 2 * R * 1.1), alpha=0.9)
                paste(img, sk.spin_circle, cx, cy, fit(sk.spin_circle, 2 * R), alpha=1.0,
                      angle=float(rng.uniform(0, 360)))
                if sk.spin_bg is None and sk.spin_circle is None:
                    cv2.circle(img, (int(cx), int(cy)), int(R), (255, 255, 255), max(2, int(R * 0.03)), cv2.LINE_AA)
            elif kind == "circle":
                x, y, col, n = dd
                self._circle(img, x, y, d, col, n, a)
            elif kind == "slider":
                pts, col, n = dd
                layer = img.copy()
                self._body(layer, pts, r, col)
                cv2.addWeighted(layer, a * 0.9, img, 1 - a * 0.9, 0, dst=img)
                if sk.slider_end is not None:                          # tail = hard negative
                    paste(img, sk.slider_end, pts[-1, 0], pts[-1, 1], d / 128.0, tint=col, alpha=a)
                else:
                    self._circle(img, pts[-1, 0], pts[-1, 1], d, col, None, a)
                if sk.reverse is not None and rng.random() < 0.3:
                    paste(img, sk.reverse, pts[-1, 0], pts[-1, 1], d / 128.0, alpha=a)
                self._circle(img, pts[0, 0], pts[0, 1], d, col, n, a, slider=True)
            elif kind == "ball":
                pts, k, col = dd
                self._body(img, pts, r, col)
                if sk.slider_end is not None:
                    paste(img, sk.slider_end, pts[-1, 0], pts[-1, 1], d / 128.0, tint=col)
                bx, by = pts[k]
                paste(img, sk.follow, bx, by, d / 128.0, alpha=float(rng.uniform(0.6, 1.0)))
                paste(img, sk.sliderb, bx, by, d / 128.0,
                      tint=sk.slider_ball_col or (col if rng.random() < 0.5 else None))
                if sk.sliderb is None:
                    cv2.circle(img, (int(bx), int(by)), int(r * 0.9), (int(col[2]), int(col[1]), int(col[0])), -1)
        for p, kind, dd in draws:                        # approach circles on top: 4x -> 1x of the circle
            a = alpha_for(p)
            if kind in ("circle", "slider"):
                x, y = (dd[0], dd[1]) if kind == "circle" else (dd[0][0, 0], dd[0][0, 1])
                col = dd[2] if kind == "circle" else dd[1]
                paste(img, sk.approach, x, y, d / 128.0 * (1 + 3 * p), tint=col, alpha=a)

        if sk.cursor is not None:                        # the skin's cursor on top (sometimes right on a note)
            if objs and rng.random() < 0.4:
                o = objs[int(rng.integers(len(objs)))]
                cx, cy = o.x + rng.normal(0, r * 0.5), o.y + rng.normal(0, r * 0.5)
            else:
                cx, cy = rng.uniform(W * 0.2, W * 0.8), rng.uniform(H * 0.2, H * 0.8)
            paste(img, sk.cursor, cx, cy, H / 768.0)          # osu! draws UI sprites at screen-height / 768

        if rng.random() < 0.4:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.0))
        f = img.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.normal(0, rng.uniform(0, 4), img.shape)
        return np.clip(f, 0, 255).astype(np.uint8), objs
