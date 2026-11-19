from __future__ import annotations
import threading
import time
from collections import deque


class SharedState:
    """Thread-safe scratchpad shared by perception, controller and visualizer."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.frame_ts = 0.0
        self.cursor = (0.0, 0.0)
        self.mode = "IDLE"
        self.path = []
        self.aim_point = None
        self.events = deque(maxlen=10)
        self.stats = {"infer_ms": 0.0, "capture_fps": 0.0, "perception_fps": 0.0}
        self.armed = False
        self.fg = ('', '')   # (process, window title) of the focused window
        self.quit = False
        self.diff = None     # (ar, od, map name) of the map being played, from diff_detect.DiffWatcher
        self.diff_ver = 0

    def set_frame(self, frame, ts):
        with self.lock:
            self.frame, self.frame_ts = frame, ts

    def get_frame(self):
        with self.lock:
            return self.frame, self.frame_ts

    def log(self, msg: str):
        with self.lock:
            self.events.appendleft(f"{time.strftime('%H:%M:%S')} {msg}")
