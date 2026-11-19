"""Record REAL training frames for the detector while you play - labelled automatically, no synthetic data.

    python record_frames.py                     # then just play maps in osu! (NoFail recommended)
    python record_frames.py --region 0,0,1920,1080 --keys "z,x"
    python train.py --real-dir data/real --no-synthetic --resume checkpoints/detector.pt --epochs 15 \
                    --out checkpoints/detector_real.pt

How the labels are made (no hand labelling):
  * while a map is on screen, ~8 screenshots per second are saved, each with its exact time, and every key press
    (your tap keys + mouse buttons) is timed too;
  * when the map ends osu! saves the replay. Lining up your key presses with the replay's key presses gives the
    exact clock offset between the screenshots and the map (same trick rl_worker.py uses);
  * the beatmap then says, for every screenshot, which circles / slider heads / spinners were on screen, where,
    how far the approach circle had closed, and where the slider ball was. Circles you already hit are left out
    (they're just fading out), exactly like the bot's tracker.
Frames go to data/real/<map>_<time>/ as .jpg + .json (the format train.py --real-dir reads).
Skipped: plays with no replay (failed without NoFail / quit), Hidden / Flashlight / speed mods, and any part of a
play after a pause (the timing would no longer line up).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from config import BotConfig, ar_to_preempt
from playfield import PlayfieldMap

LOCAL = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "osu!"
MAP_TITLE = re.compile(r"^osu!.*?\s-\s+(.+)$")
MOD_HD, MOD_FL = 8, 1024
OUT = Path("data/real")
TMP = Path("data/real_tmp")


# ====================================================================== key timing (Windows)
def vk_for(key: str):
    k = key.strip().lower()
    if k in ("mouse1", "m1", "lmb"):
        return 0x01
    if k in ("mouse2", "m2", "rmb"):
        return 0x02
    if platform.system() != "Windows" or not k:
        return None
    import ctypes
    v = ctypes.windll.user32.VkKeyScanW(ord(k[0]))
    return None if v == -1 else (v & 0xFF)


class KeyLogger(threading.Thread):
    """Polls the physical tap keys at ~1 kHz; records the time of every key-DOWN."""

    def __init__(self, keys):
        super().__init__(daemon=True)
        self.vks = [v for v in (vk_for(k) for k in keys) if v is not None]
        self.presses: list[float] = []
        self.lock = threading.Lock()
        self._stop_flag = False

    def run(self):
        if platform.system() != "Windows":
            return
        import ctypes
        u = ctypes.windll.user32
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:  # noqa
            pass
        prev = {v: False for v in self.vks}
        while not self._stop_flag:
            t = time.perf_counter()
            any_new = False
            for v in self.vks:
                down = bool(u.GetAsyncKeyState(v) & 0x8000)
                if down and not prev[v]:
                    any_new = True
                prev[v] = down
            if any_new:
                with self.lock:
                    if not self.presses or t - self.presses[-1] > 1e-3:
                        self.presses.append(t)
            time.sleep(0.0005)

    def take(self, t0, t1):
        with self.lock:
            return [p for p in self.presses if t0 <= p <= t1]

    def stop(self):
        self._stop_flag = True


# ====================================================================== labelling one play
def label_frames(bm, rep, frame_times_map, pfm: PlayfieldMap, frame_w: int, frame_h: int):
    """-> list (per frame) of label dicts in train.py's --real-dir format (x, y, radius normalised)."""
    from judge import hit_windows
    from osu_formats import SLIDER, SPINNER
    from replay_data import judge_replay
    judge = judge_replay(bm, rep)
    preempt = ar_to_preempt(bm.ar)
    _, _, w50 = hit_windows(bm.od)
    r_frame = bm.radius * pfm.s
    out = []
    for t in frame_times_map:
        objs = []
        for o in bm.objects:
            if o.kind == SPINNER:
                if o.time - 0.15 <= t <= o.end_time:
                    fx, fy = pfm.pf_to_frame(256.0, 192.0)
                    prog = float(np.clip((o.end_time - t) / max(o.end_time - o.time, 1e-3), 0, 1))
                    objs.append({"cls": 2, "x": fx / frame_w, "y": fy / frame_h, "progress": prog,
                                 "radius": 0.33 * frame_h / frame_w})
                continue
            gone = judge.resolved.get(o.index, o.time + w50)      # hit -> it explodes / fades: not a target any more
            if o.time - preempt <= t < gone:
                fx, fy = pfm.pf_to_frame(o.x, o.y)
                objs.append({"cls": 1 if o.kind == SLIDER else 0, "x": fx / frame_w, "y": fy / frame_h,
                             "progress": float(np.clip((o.time - t) / preempt, 0.0, 1.0)),
                             "radius": r_frame / frame_w})
            if o.kind == SLIDER and o.time <= t <= o.end_time:
                b = o.ball_pos(t)
                fx, fy = pfm.pf_to_frame(float(b[0]), float(b[1]))
                objs.append({"cls": 3, "x": fx / frame_w, "y": fy / frame_h, "progress": 0.0,
                             "radius": r_frame / frame_w})
        out.append([o for o in objs if 0 <= o["x"] <= 1 and 0 <= o["y"] <= 1])
    return out


def trusted_frames(frame_t, live_press, offset, rep_press, tol=0.015, reach=5.0):
    """Keep frames whose time mapping is confirmed by nearby key presses (drops everything after a pause)."""
    if len(live_press) == 0:
        return np.zeros(len(frame_t), bool)
    lp = np.sort(np.asarray(live_press))
    rp = np.sort(np.asarray(rep_press))
    s = lp + offset
    j = np.clip(np.searchsorted(rp, s), 1, len(rp) - 1)
    dev = np.minimum(np.abs(rp[j] - s), np.abs(rp[j - 1] - s))
    good = dev < tol
    keep = np.zeros(len(frame_t), bool)
    for i, t in enumerate(frame_t):
        k = np.searchsorted(lp, t)
        prev_ok = k > 0 and t - lp[k - 1] < reach and good[k - 1]
        next_ok = k < len(lp) and lp[k] - t < reach and good[k]
        keep[i] = (prev_ok or k == 0) and (next_ok or k == len(lp)) and (prev_ok or next_ok)
    return keep


# ====================================================================== recorder
class Recorder:
    def __init__(self, a):
        self.a = a
        self.cfg = BotConfig()
        l, t, w, h = (int(v) for v in a.region.split(","))
        c = self.cfg.capture
        c.left, c.top, c.width, c.height = l, t, w, h
        self.cfg.playfield.y_shift = a.pf_shift
        self.pfm = PlayfieldMap(c, self.cfg.playfield)
        self.keys = KeyLogger(a.keys.split(",") + ["mouse1", "mouse2"])
        self.idx = None

    def log(self, m):
        print(f"[rec] {m}", flush=True)

    def _beatmap(self, md5):
        from replay_data import index_songs
        if self.idx is None or md5 not in self.idx:
            self.idx = {}
            for d in [self.a.songs, "songs"]:
                if d and os.path.isdir(d):
                    self.idx.update(index_songs(d))
        return self.idx.get(md5)

    def _newest_replay_after(self, t_wall):
        best = None
        for d in self.a.replay_dirs:
            if not os.path.isdir(d):
                continue
            for p in Path(d).glob("*.osr"):
                try:
                    m = p.stat().st_mtime
                except OSError:
                    continue
                if m >= t_wall - 2 and (best is None or m > best[0]):
                    best = (m, p)
        return best[1] if best else None

    def record_play(self, sct, region, name):
        """Capture frames while the map is on screen. -> (session dir, frame times, wall start, perf start, perf end)"""
        from inputs import foreground_info
        sess = TMP / time.strftime("%Y%m%d_%H%M%S")
        sess.mkdir(parents=True, exist_ok=True)
        times, period = [], 1.0 / self.a.fps
        wall0, t_start = time.time(), time.perf_counter()
        last_title_check, in_map = 0.0, True
        W, H = self.cfg.model.input_w, self.cfg.model.input_h
        while in_map:
            t0 = time.perf_counter()
            shot = np.asarray(sct.grab(region))[:, :, :3]
            ts = time.perf_counter() - self.cfg.capture.display_latency
            small = cv2.resize(shot, (W, H), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                buf.tofile(str(sess / f"{len(times):06d}.jpg"))
                times.append(ts)
            if t0 - last_title_check > 0.25:
                last_title_check = t0
                proc, title = foreground_info()
                if proc and proc.startswith("osu"):
                    in_map = bool(MAP_TITLE.match(title or ""))
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
        return sess, np.array(times), wall0, t_start, time.perf_counter()

    def finish(self, sess, ftimes, wall0, p0, p1, name):
        from osu_formats import parse_osr, parse_osu, BAD_MODS
        from rl_worker import align, replay_presses
        self.log(f"map over ({len(ftimes)} frames) - waiting for osu! to save the replay...")
        rp_path, deadline = None, time.time() + self.a.wait_replay
        while time.time() < deadline:
            rp_path = self._newest_replay_after(wall0)
            if rp_path is not None and time.time() - rp_path.stat().st_mtime > 1.0:
                break
            rp_path = None
            time.sleep(1.0)
        if rp_path is None:
            self.log("no replay saved (failed or quit? use NoFail) - frames discarded")
            return shutil.rmtree(sess, ignore_errors=True)
        rep = parse_osr(rp_path)
        if rep.mode != 0 or rep.mods & (BAD_MODS | MOD_HD | MOD_FL):
            self.log(f"replay uses mods {rep.mods} (speed / Hidden / Flashlight) - frames discarded")
            return shutil.rmtree(sess, ignore_errors=True)
        osu = self._beatmap(rep.beatmap_md5)
        if osu is None:
            self.log("this map's .osu isn't in your Songs folder - frames discarded")
            return shutil.rmtree(sess, ignore_errors=True)
        bm = parse_osu(osu).with_mods(rep.mods)
        live = np.array(self.keys.take(p0 - 1.0, p1 + 1.0))
        rpp = replay_presses(rep)[:, 0]
        res = align(live, rpp)
        if res is None or res[1] < max(5, 0.6 * len(live)):
            self.log(f"couldn't line up key presses with the replay ({0 if res is None else res[1]}/{len(live)} matched)"
                     " - frames discarded. Are --keys your osu! tap keys?")
            return shutil.rmtree(sess, ignore_errors=True)
        offset, n, frac, resid = res
        keep = trusted_frames(ftimes, live, offset, rpp)
        t_map = ftimes + offset
        first = bm.objects[0].time - ar_to_preempt(bm.ar) - 0.2
        last = max(o.end_time for o in bm.objects) + 0.3
        keep &= (t_map >= first) & (t_map <= last)
        labels = label_frames(bm, rep, t_map, self.pfm, self.cfg.capture.width, self.cfg.capture.height)
        safe = "".join(ch if ch.isalnum() or ch in " -_" else "_" for ch in (bm.title or name))[:50].strip()
        out = OUT / f"{safe}_{sess.name}"
        out.mkdir(parents=True, exist_ok=True)
        saved = 0
        for i in np.nonzero(keep)[0]:
            src = sess / f"{i:06d}.jpg"
            if not src.exists():
                continue
            os.replace(src, out / src.name)
            (out / f"{i:06d}.json").write_text(json.dumps({"objects": labels[i]}))
            saved += 1
        shutil.rmtree(sess, ignore_errors=True)
        n_obj = sum(len(labels[i]) for i in np.nonzero(keep)[0])
        self.log(f"saved {saved} labelled frames ({n_obj} objects) -> {out}  "
                 f"[{n}/{len(live)} presses matched, +/-{resid:.1f} ms]")
        total = sum(1 for _ in OUT.rglob("*.jpg"))
        self.log(f"total real frames so far: {total}")

    def run(self):
        import mss
        from inputs import foreground_info
        self.keys.start()
        c = self.cfg.capture
        region = {"left": c.left, "top": c.top, "width": c.width, "height": c.height}
        self.log(f"recording region {c.width}x{c.height} at {self.a.fps} fps while a map is on screen. "
                 f"Tap keys: {self.a.keys} + mouse buttons. Play maps (NoFail); Ctrl+C to stop.")
        with mss.mss() as sct:
            while True:
                proc, title = foreground_info()
                m = MAP_TITLE.match(title or "") if proc and proc.startswith("osu") else None
                if not m:
                    time.sleep(0.2)
                    continue
                self.log(f"map started: {m.group(1)[:60]}")
                sess, ft, wall0, p0, p1 = self.record_play(sct, region, m.group(1))
                if len(ft) < 20:
                    shutil.rmtree(sess, ignore_errors=True)
                    continue
                threading.Thread(target=self.finish, args=(sess, ft, wall0, p0, p1, m.group(1)), daemon=True).start()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default="0,0,1920,1080", help="left,top,width,height of the osu! window (same as run_bot)")
    ap.add_argument("--keys", default="z,x,[,]",
                    help="keys you tap with in osu!, comma-separated (mouse buttons are always included)")
    ap.add_argument("--fps", type=float, default=8.0, help="screenshots per second while a map is playing")
    ap.add_argument("--pf-shift", type=float, default=0.0)
    ap.add_argument("--songs", default=str(LOCAL / "Songs"))
    ap.add_argument("--replay-dirs", nargs="+", default=[str(LOCAL / "Data" / "r"), str(LOCAL / "Replays")])
    ap.add_argument("--wait-replay", type=float, default=60.0)
    a = ap.parse_args()
    if platform.system() != "Windows":
        raise SystemExit("record_frames.py needs Windows (screen capture + key timing).")
    try:
        Recorder(a).run()
    except KeyboardInterrupt:
        print("\n[rec] stopped")


if __name__ == "__main__":
    main()
