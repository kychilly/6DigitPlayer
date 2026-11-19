"""Auto-detect the current map's AR / OD (like a human reading song select) so the bot doesn't need --ar/--od.

osu!stable's window title while playing is "osu!  - Artist - Title [Difficulty]". We look that up in a small
index of the Songs folder (only the [Metadata] / [Difficulty] headers are read - never the notes) and hand AR/OD
to the tracker + policy controller. AR decides how fast approach circles close (-> time-to-hit), OD decides the
300/100/50 windows, so getting them right matters for both playing and learning.

Mods are NOT visible in the title: with HR / EZ / DT the real AR/OD differ (rl_worker.py warns after the map).
osu!lazer doesn't put the map in its title, so there the --ar/--od values are used.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

from config import ar_to_preempt
from inputs import foreground_info

CACHE = Path("data/song_meta_cache.json")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def read_header(path: Path) -> dict | None:
    meta, sec, ar = {}, "", None
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("["):
                sec = line.strip("[]")
                if sec in ("TimingPoints", "HitObjects", "Events", "Colours"):
                    if "od" in meta:
                        break
                continue
            if ":" not in line:
                continue
            k, v = (s.strip() for s in line.split(":", 1))
            if sec == "Metadata" and k in ("Artist", "Title", "Version"):
                meta[k.lower()] = v
            elif sec == "Difficulty":
                try:
                    if k == "OverallDifficulty":
                        meta["od"] = float(v)
                    elif k == "ApproachRate":
                        ar = float(v)
                    elif k == "CircleSize":
                        meta["cs"] = float(v)
                except ValueError:
                    pass
    if "od" not in meta or "title" not in meta:
        return None
    meta["ar"] = ar if ar is not None else meta["od"]          # very old maps: AR == OD
    return meta


class SongMetaIndex:
    """'artist - title [version]' -> {ar, od, cs}. Built once in the background, cached by file mtime."""

    def __init__(self, songs_dir: str, log=print):
        self.songs_dir, self.log = songs_dir, log
        self.by_key: dict[str, dict] = {}
        self.ready = False

    def build(self):
        cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        out, n_new = {}, 0
        root = Path(self.songs_dir)
        if not root.is_dir():
            self.log(f"[diff] Songs folder not found ({self.songs_dir}); using --ar/--od")
            self.ready = True
            return
        for p in root.rglob("*.osu"):
            sp = str(p)
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            c = cache.get(sp)
            if c is None or c.get("mtime") != m:
                h = read_header(p)
                if h is None:
                    continue
                c = {**h, "mtime": m}
                n_new += 1
            out[sp] = c
            key = norm(f"{c.get('artist', '')} - {c['title']} [{c.get('version', '')}]")
            self.by_key[key] = c
        if n_new:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(out))
        self.ready = True
        self.log(f"[diff] indexed {len(self.by_key)} difficulties ({n_new} new) for AR/OD auto-detect")

    def lookup(self, window_title: str):
        m = re.match(r"^osu!.*?\s-\s+(.+)$", window_title or "")
        if not m:
            return None, None
        name = m.group(1).strip()
        return self.by_key.get(norm(name)), name


class DiffWatcher(threading.Thread):
    """Polls the osu! window title; when a new map starts, publishes its AR/OD to shared.diff."""

    def __init__(self, songs_dir: str, shared, tracker=None, log=print, period: float = 0.25):
        super().__init__(daemon=True)
        self.index = SongMetaIndex(songs_dir, log)
        self.shared, self.tracker, self.log, self.period = shared, tracker, log, period
        self._last_title = None
        self._stop_flag = False

    def run(self):
        try:
            self.index.build()
        except Exception as e:  # noqa
            self.log(f"[diff] index failed ({e}); using --ar/--od")
            return
        while not self._stop_flag:
            time.sleep(self.period)
            proc, title = foreground_info()
            if not proc or not proc.startswith("osu") or title == self._last_title:
                continue
            self._last_title = title
            meta, name = self.index.lookup(title)
            if name is None:
                continue                                  # menus / song select
            if meta is None:
                self.shared.log(f"map '{name[:40]}' not in Songs index - keeping AR/OD as they were")
                continue
            if self.tracker is not None:
                self.tracker.set_preempt(ar_to_preempt(meta["ar"]))
            with self.shared.lock:
                self.shared.diff = (meta["ar"], meta["od"], name)
                self.shared.diff_ver += 1

    def stop(self):
        self._stop_flag = True
