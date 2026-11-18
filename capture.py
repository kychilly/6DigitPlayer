from __future__ import annotations
import threading
import time
import mss
import numpy as np
from config import CaptureConfig


class ScreenCapture(threading.Thread):
    """Grabs the osu! window region as fast as possible; always keeps the newest frame."""

    def __init__(self, cfg: CaptureConfig):
        super().__init__(daemon=True)
        self.cfg = cfg
        self._lock = threading.Lock()
        self._latest = None
        self._seq = 0
        self._stop_flag = False
        self.fps = 0.0

    def run(self):
        region = {"left": self.cfg.left, "top": self.cfg.top,
                  "width": self.cfg.width, "height": self.cfg.height}
        period = 1.0 / self.cfg.fps
        n, t_fps = 0, time.perf_counter()
        with mss.mss() as sct:  # mss handles are thread-local on Windows
            while not self._stop_flag:
                t0 = time.perf_counter()
                shot = sct.grab(region)
                ts = time.perf_counter() - self.cfg.display_latency
                frame = np.asarray(shot)[:, :, :3]  # BGRA -> BGR
                with self._lock:
                    self._latest = (ts, frame)
                    self._seq += 1
                n += 1
                if t0 - t_fps > 1.0:
                    self.fps, n, t_fps = n / (t0 - t_fps), 0, t0
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def latest(self):
        """-> (seq, timestamp, BGR frame) or (0, 0, None)."""
        with self._lock:
            if self._latest is None:
                return 0, 0.0, None
            return self._seq, self._latest[0], self._latest[1]

    def stop(self):
        self._stop_flag = True
