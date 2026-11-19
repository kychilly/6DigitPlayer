"""Tells the bot the moment a map is over, so the play is saved for learning right away:
  * osu! wrote a new replay (that happens when the results screen appears), or
  * the osu! window title left the map (failed / quit back to song select).
Without this the bot would wait for `episode_gap` seconds with no notes before closing the episode."""
from __future__ import annotations

import os
import re
import threading
import time

from inputs import foreground_info

MAP_TITLE = re.compile(r"^osu!.*?\s-\s+.+")      # "osu!  - Artist - Title [Diff]" while a map is loaded


class MapEndWatcher(threading.Thread):
    def __init__(self, shared, replay_dirs, period: float = 0.3):
        super().__init__(daemon=True)
        self.shared, self.dirs, self.period = shared, [d for d in replay_dirs if d], period
        self._stop_flag = False

    def _dir_stamp(self):
        """Cheap: a folder's mtime changes when a file is created in it (no need to list thousands of replays)."""
        out = []
        for d in self.dirs:
            try:
                out.append(os.stat(d).st_mtime_ns)
            except OSError:
                out.append(None)
        return out

    def _signal(self, reason: str):
        with self.shared.lock:
            self.shared.map_end_ver += 1
            self.shared.map_end_reason = reason

    def run(self):
        stamp = self._dir_stamp()
        in_map = False
        while not self._stop_flag:
            time.sleep(self.period)
            s = self._dir_stamp()
            if s != stamp:
                stamp = s
                self._signal("replay saved (results screen)")
            proc, title = foreground_info()
            if proc and proc.startswith("osu"):
                now_in_map = bool(MAP_TITLE.match(title or ""))
                if in_map and not now_in_map:
                    self._signal("left the map")
                in_map = now_in_map

    def stop(self):
        self._stop_flag = True
