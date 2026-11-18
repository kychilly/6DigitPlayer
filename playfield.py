"""Playfield <-> screen coordinate mapping (osu! draws its 512x384 playfield at 80% of screen height)."""
from __future__ import annotations
from config import CaptureConfig, PlayfieldConfig


class PlayfieldMap:
    def __init__(self, cap: CaptureConfig, pf: PlayfieldConfig):
        H, W = cap.height, cap.width
        self.s = pf.scale or H / 480.0
        self.x0 = pf.x0 or (W - 512.0 * self.s) / 2.0
        self.y0 = pf.y0 or ((H - 384.0 * self.s) / 2.0 + pf.y_shift * self.s)
        self.left, self.top = cap.left, cap.top

    # capture-frame pixels <-> playfield
    def frame_to_pf(self, x, y):
        return (x - self.x0) / self.s, (y - self.y0) / self.s

    def pf_to_frame(self, x, y):
        return self.x0 + x * self.s, self.y0 + y * self.s

    # absolute screen pixels <-> playfield
    def screen_to_pf(self, x, y):
        return (x - self.left - self.x0) / self.s, (y - self.top - self.y0) / self.s

    def pf_to_screen(self, x, y):
        return self.left + self.x0 + x * self.s, self.top + self.y0 + y * self.s

    def rect_frame(self):
        return self.x0, self.y0, self.x0 + 512 * self.s, self.y0 + 384 * self.s
