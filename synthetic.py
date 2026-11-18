"""Procedural osu!-like frame generator with exact labels, so the detector can be
bootstrapped without hand-labelling. Randomises backgrounds, colours, circle sizes,
fade-in, sliders (with hard-negative tail circles), slider balls, spinners, HUD clutter."""
from __future__ import annotations
import math
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class GTObject:
    cls: int
    x: float
    y: float
    progress: float
    radius: float


class SyntheticRenderer:
    def __init__(self, W=384, H=224, stride=4, num_classes=4):
        self.W, self.H, self.stride, self.C = W, H, stride, num_classes

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _blend(img, fn, alpha):
        if alpha >= 0.99:
            fn(img)
            return
        layer = img.copy()
        fn(layer)
        cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, dst=img)

    def _background(self, rng):
        W, H = self.W, self.H
        kind = rng.integers(0, 4)
        if kind == 0:
            img = np.full((H, W, 3), rng.integers(0, 90, 3), np.uint8)
        elif kind == 1:
            a, b = rng.integers(0, 120, 3), rng.integers(0, 120, 3)
            t = np.linspace(0, 1, W)[None, :, None]
            if rng.random() < 0.5:
                t = np.linspace(0, 1, H)[:, None, None]
            img = (a * (1 - t) + b * t) * np.ones((H, W, 1))
            img = img.astype(np.uint8)
        elif kind == 2:
            small = rng.integers(0, 140, (rng.integers(3, 9), rng.integers(4, 12), 3)).astype(np.uint8)
            img = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
        else:
            img = np.full((H, W, 3), rng.integers(10, 70, 3), np.uint8)
            for _ in range(rng.integers(3, 12)):
                p1 = (int(rng.integers(0, W)), int(rng.integers(0, H)))
                p2 = (int(rng.integers(0, W)), int(rng.integers(0, H)))
                cv2.rectangle(img, p1, p2, tuple(int(v) for v in rng.integers(0, 150, 3)), -1)
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(2, 8))
        return (img.astype(np.float32) * rng.uniform(0.35, 1.0)).astype(np.uint8)

    @staticmethod
    def _color(rng):
        hsv = np.array([[[rng.integers(0, 180), rng.integers(120, 255), rng.integers(150, 255)]]], np.uint8)
        return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])

    def _hitcircle(self, img, x, y, r, color, number, alpha, border):
        def fn(im):
            c = (int(x), int(y))
            cv2.circle(im, c, int(r), color, -1, cv2.LINE_AA)
            cv2.circle(im, c, max(1, int(r * 0.82)), tuple(int(v * 0.55) for v in color), -1, cv2.LINE_AA)
            cv2.circle(im, c, int(r), border, max(1, int(round(r * 0.12))), cv2.LINE_AA)
            if number is not None:
                s = max(0.25, r / 22.0)
                (tw, th), _ = cv2.getTextSize(str(number), cv2.FONT_HERSHEY_SIMPLEX, s, 1)
                cv2.putText(im, str(number), (int(x - tw / 2), int(y + th / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 1, cv2.LINE_AA)
        self._blend(img, fn, alpha)

    def _ring(self, img, x, y, r, p, alpha, color):
        rr = int(r * (1 + 3 * p))
        if rr <= int(r) + 1:
            return
        self._blend(img, lambda im: cv2.circle(im, (int(x), int(y)), rr, color,
                                               max(1, int(r * 0.06)), cv2.LINE_AA), alpha)

    def _slider_path(self, rng, x, y, r):
        W, H = self.W, self.H
        L = r * rng.uniform(3, 8)
        ang = rng.uniform(0, 2 * math.pi)
        ex = float(np.clip(x + L * math.cos(ang), W * 0.2, W * 0.8))
        ey = float(np.clip(y + L * math.sin(ang), H * 0.12, H * 0.88))
        cx = (x + ex) / 2 + rng.normal(0, r * 2)
        cy = (y + ey) / 2 + rng.normal(0, r * 2)
        t = np.linspace(0, 1, 28)[:, None]
        p0, p1, p2 = np.array([x, y]), np.array([cx, cy]), np.array([ex, ey])
        return ((1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2)

    def _slider_body(self, img, pts, r, color, border):
        for rad, col in ((r, border), (r * 0.88, (28, 28, 28))):
            th = max(1, int(2 * rad))
            ip = pts.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [ip], False, col, th, cv2.LINE_AA)
            for p in ip[:, 0]:
                cv2.circle(img, (int(p[0]), int(p[1])), int(rad), col, -1, cv2.LINE_AA)

    # ------------------------------------------------------------ render
    def render(self, rng):
        W, H = self.W, self.H
        img = self._background(rng)
        objs: list[GTObject] = []
        r_base = rng.uniform(9, 20)
        border = (255, 255, 255) if rng.random() < 0.85 else self._color(rng)
        ring_col = (255, 255, 255) if rng.random() < 0.8 else self._color(rng)
        taken = []

        def free_pos(r, tries=12, allow_stack=False):
            for _ in range(tries):
                x = rng.uniform(W * 0.24, W * 0.76)
                y = rng.uniform(H * 0.18, H * 0.82)
                if allow_stack and taken and rng.random() < 0.5:
                    x, y = taken[rng.integers(len(taken))]
                    return x, y
                if all(math.hypot(x - a, y - b) > 2.3 * r for a, b in taken):
                    return x, y
            return None

        draws = []  # (progress, kind, data)
        menu = rng.random() < 0.12          # main-menu frame: giant logo, NO objects
        spinner = (not menu) and rng.random() < 0.05
        if spinner:
            R = rng.uniform(0.28, 0.38) * H
            cx, cy = W / 2 + rng.normal(0, 3), H / 2 + rng.normal(0, 3)
            p = rng.uniform(0.0, 1.0)
            draws.append((p, "spinner", (cx, cy, R)))
            objs.append(GTObject(2, cx, cy, p, R))

        n_circ = 0 if (spinner or menu) else int(rng.integers(0, 6))
        n_slid = 0 if (spinner or menu) else int(rng.integers(0, 3))
        n_ball = 1 if (not (spinner or menu) and rng.random() < 0.35) else 0
        used_p = []
        for i in range(n_circ):
            pos = free_pos(r_base, allow_stack=rng.random() < 0.1)
            if pos is None:
                continue
            p = rng.uniform(0.02, 1.0)
            if any(abs(p - q) < 0.08 for q in used_p):
                continue
            used_p.append(p)
            taken.append(pos)
            draws.append((p, "circle", (pos[0], pos[1], self._color(rng), i + 1)))
            objs.append(GTObject(0, pos[0], pos[1], p, r_base))
        for i in range(n_slid + n_ball):
            pos = free_pos(r_base * 1.2)
            if pos is None:
                continue
            taken.append(pos)
            pts = self._slider_path(rng, pos[0], pos[1], r_base)
            for q in pts[::4]:
                taken.append((q[0], q[1]))
            if i < n_slid:
                p = rng.uniform(0.02, 1.0)
                draws.append((p, "slider", (pts, self._color(rng), i + 1)))
                objs.append(GTObject(1, pts[0, 0], pts[0, 1], p, r_base))
            else:
                k = int(rng.integers(1, len(pts) - 1))
                draws.append((0.0, "ball", (pts, k, self._color(rng))))
                objs.append(GTObject(3, pts[k, 0], pts[k, 1], 0.0, r_base))

        # distractors (no labels)
        self._clutter(img, rng, r_base)
        if menu:
            self._logo(img, rng)

        # draw: bodies first (later objects underneath), then circles, then approach rings
        for p, kind, d in sorted(draws, key=lambda t: -t[0]):
            alpha = float(np.clip(1.3 * (1 - p) + 0.25, 0.3, 1.0))
            if kind == "spinner":
                cx, cy, R = d
                self._blend(img, lambda im: cv2.circle(im, (int(cx), int(cy)), int(R * 0.35), (255, 255, 255), 2, cv2.LINE_AA), 0.5)
                cv2.circle(img, (int(cx), int(cy)), int(R), (255, 255, 255), max(2, int(R * 0.03)), cv2.LINE_AA)
            elif kind == "circle":
                x, y, col, n = d
                self._hitcircle(img, x, y, r_base, col, n, alpha, border)
            elif kind == "slider":
                pts, col, n = d
                self._blend(img, lambda im: self._slider_body(im, pts, r_base, col, border), alpha)
                self._hitcircle(img, pts[-1, 0], pts[-1, 1], r_base, col, None, alpha, border)  # tail = hard negative
                self._hitcircle(img, pts[0, 0], pts[0, 1], r_base, col, n, alpha, border)
            elif kind == "ball":
                pts, k, col = d
                self._slider_body(img, pts, r_base, col, border)
                self._hitcircle(img, pts[-1, 0], pts[-1, 1], r_base, col, None, 1.0, border)
                bx, by = pts[k]
                self._blend(img, lambda im: cv2.circle(im, (int(bx), int(by)), int(r_base * 2.4), (255, 255, 255), -1, cv2.LINE_AA), 0.15)
                cv2.circle(img, (int(bx), int(by)), int(r_base * 2.4), (255, 255, 255), max(1, int(r_base * 0.07)), cv2.LINE_AA)
                cv2.circle(img, (int(bx), int(by)), int(r_base * 0.9), col, -1, cv2.LINE_AA)
                cv2.circle(img, (int(bx), int(by)), int(r_base * 0.9), (255, 255, 255), 1, cv2.LINE_AA)
        for p, kind, d in draws:
            alpha = float(np.clip(1.3 * (1 - p) + 0.25, 0.3, 1.0))
            if kind == "circle":
                self._ring(img, d[0], d[1], r_base, p, alpha, ring_col)
            elif kind == "slider":
                self._ring(img, d[0][0, 0], d[0][0, 1], r_base, p, alpha, ring_col)
            elif kind == "spinner":
                cx, cy, R = d
                cv2.circle(img, (int(cx), int(cy)), max(2, int(R * (0.2 + 0.8 * p))), (255, 255, 255), 2, cv2.LINE_AA)

        # cursor on top
        cx, cy = int(rng.uniform(W * 0.2, W * 0.8)), int(rng.uniform(H * 0.2, H * 0.8))
        for k in range(6):
            cv2.circle(img, (cx - k * 4, cy + k * 2), max(1, 4 - k // 2), (200, 200, 200), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)

        # camera-ish post-processing
        if rng.random() < 0.4:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.2))
        f = img.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.normal(0, rng.uniform(0, 5), img.shape)
        return np.clip(f, 0, 255).astype(np.uint8), objs

    def _logo(self, img, rng):
        """Main-menu logo: huge white-rimmed circle with 'osu!' text. A hard negative."""
        W, H = self.W, self.H
        R = int(rng.uniform(0.28, 0.42) * H)
        cx, cy = int(W / 2 + rng.normal(0, W * 0.03)), int(H / 2 + rng.normal(0, H * 0.03))
        col = (150, 80, 235) if rng.random() < 0.6 else self._color(rng)
        cv2.circle(img, (cx, cy), R, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), int(R * 0.9), col, -1, cv2.LINE_AA)
        s = R / 30.0
        (tw, th), _ = cv2.getTextSize("osu!", cv2.FONT_HERSHEY_SIMPLEX, s, 4)
        cv2.putText(img, "osu!", (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 4, cv2.LINE_AA)
        for y0, y1 in ((0, int(H * 0.11)), (int(H * 0.89), H)):
            self._blend(img, lambda im: cv2.rectangle(im, (0, y0), (W, y1), (0, 0, 0), -1), 0.4)

    def _clutter(self, img, rng, r):
        W, H = self.W, self.H
        for _ in range(rng.integers(0, 4)):  # hit bursts / judgement text
            x, y = int(rng.uniform(W * 0.2, W * 0.8)), int(rng.uniform(H * 0.2, H * 0.8))
            col = self._color(rng)
            self._blend(img, lambda im: cv2.circle(im, (x, y), int(r * rng.uniform(1.1, 1.8)), col, max(1, int(r * 0.1)), cv2.LINE_AA), 0.35)
            cv2.putText(img, rng.choice(["300", "100", "50", "x"]), (x - 8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
        cv2.rectangle(img, (4, 4), (int(W * rng.uniform(0.2, 0.5)), 10), (60, 220, 90), -1)
        cv2.putText(img, "%08d" % rng.integers(0, 10 ** 8), (W - 90, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "%dx" % rng.integers(0, 999), (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ------------------------------------------------------------ targets
    def build_targets(self, objs):
        s = self.stride
        h, w = self.H // s, self.W // s
        heat = np.zeros((self.C, h, w), np.float32)
        reg = np.zeros((4, h, w), np.float32)
        mask = np.zeros((1, h, w), np.float32)
        for o in objs:
            cx, cy = o.x / s, o.y / s
            ix, iy = int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1))
            sigma = max(0.8, (min(o.radius, self.H * 0.15) / s) / 2.5)
            rad = int(math.ceil(3 * sigma))
            y0, y1, x0, x1 = max(0, iy - rad), min(h, iy + rad + 1), max(0, ix - rad), min(w, ix + rad + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            g = np.exp(-((xx - ix) ** 2 + (yy - iy) ** 2) / (2 * sigma ** 2))
            heat[o.cls, y0:y1, x0:x1] = np.maximum(heat[o.cls, y0:y1, x0:x1], g)
            reg[:, iy, ix] = [cx - ix, cy - iy, o.progress, o.radius / 32.0]
            mask[0, iy, ix] = 1.0
        return heat, reg, mask
