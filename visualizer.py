"""Real-time debug view: captured frame + everything the bot currently 'knows'."""
from __future__ import annotations
import time
import cv2
import numpy as np
from config import CLASS_NAMES
from playfield import PlayfieldMap

COLORS = {0: (80, 255, 80), 1: (255, 255, 0), 2: (255, 80, 255), 3: (0, 220, 255)}


class Visualizer:
    def __init__(self, cfg, tracker, audio, shared, disp_w=960, panel_w=360):
        self.cfg, self.tracker, self.audio, self.shared = cfg, tracker, audio, shared
        self.disp_w, self.panel_w = disp_w, panel_w
        self.pfm = PlayfieldMap(cfg.capture, cfg.playfield)

    def render(self):
        frame, _ = self.shared.get_frame()
        now = time.perf_counter()
        if frame is None:
            canvas = np.zeros((540, self.disp_w + self.panel_w, 3), np.uint8)
            cv2.putText(canvas, "waiting for frames...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            return canvas
        sc = self.disp_w / frame.shape[1]
        view = cv2.resize(frame, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        x0, y0, x1, y1 = self.pfm.rect_frame()
        cv2.rectangle(view, (int(x0 * sc), int(y0 * sc)), (int(x1 * sc), int(y1 * sc)), (0, 160, 0), 1)  # playfield box
        objs = self.tracker.snapshot()
        pend = [o for o in objs if o.hits >= self.cfg.tracker.min_hits and not o.active and not o.bad]
        order = {o.id: i for i, o in enumerate(pend)}
        for o in objs:
            col = (110, 110, 110) if o.bad else COLORS[o.cls]
            ok = o.hits >= self.cfg.tracker.min_hits
            c = (int(o.x * sc), int(o.y * sc))
            cv2.circle(view, c, max(3, int(o.radius * sc)), col, 2 if ok else 1, cv2.LINE_AA)
            ttp = (o.hit_time - now) * 1000
            lab = f"{order.get(o.id, '-')}:#{o.id} {ttp:+.0f}ms" + (" STATIC-ignored" if o.bad else "*" if o.verified else "")
            cv2.putText(view, lab, (c[0] + 6, c[1] - int(o.radius * sc) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        ball = self.tracker.ball()
        if ball and now - ball.t < 0.3:
            cv2.drawMarker(view, (int(ball.x * sc), int(ball.y * sc)), COLORS[3], cv2.MARKER_DIAMOND, 18, 2)

        ox, oy = self.cfg.capture.left, self.cfg.capture.top
        with self.shared.lock:
            path, aim, cur = list(self.shared.path), self.shared.aim_point, self.shared.cursor
            events, stats, mode, armed = list(self.shared.events), dict(self.shared.stats), self.shared.mode, self.shared.armed
        if len(path) > 1:
            pts = np.array([[(p[0] - ox) * sc, (p[1] - oy) * sc] for p in path], np.int32).reshape(-1, 1, 2)
            cv2.polylines(view, [pts], False, (0, 165, 255), 1, cv2.LINE_AA)
        if aim:
            cv2.drawMarker(view, (int((aim[0] - ox) * sc), int((aim[1] - oy) * sc)), (0, 165, 255), cv2.MARKER_CROSS, 12, 1)
        cv2.circle(view, (int((cur[0] - ox) * sc), int((cur[1] - oy) * sc)), 5, (0, 0, 255), -1, cv2.LINE_AA)

        panel = np.zeros((view.shape[0], self.panel_w, 3), np.uint8)
        y = 22

        def line(t, col=(230, 230, 230)):
            nonlocal y
            cv2.putText(panel, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            y += 18

        if armed:
            line("ARMED - " + ("waiting for osu! map" if mode == "STANDBY" else "CONTROLLING MOUSE"), (0, 0, 255))
        else:
            line("DISARMED  [F8 / SPACE] arm   [ESC] kill", (0, 200, 255))
        line(f"mode {mode}  cap {stats['capture_fps']:.0f}fps  net {stats['perception_fps']:.0f}fps ({stats['infer_ms']:.1f}ms)")
        proc, title = self.shared.fg
        line(f"focus: {proc or '?'} | {title[:34]}", (170, 170, 170))
        if self.audio is not None:
            ai = self.audio.info()
            if ai["available"]:
                bpm = f"{ai['bpm']:.0f}bpm conf {ai['conf']:.2f}" if ai["bpm"] else "tempo ?"
                flash = ai["since_onset"] is not None and ai["since_onset"] < 0.06
                line(f"audio: {bpm}", (80, 255, 80) if flash else (200, 200, 200))
            else:
                line("audio: off", (120, 120, 120))
        if "judge" in stats:
            line(f"judge 300/100/50/miss: {stats['judge']}", (120, 255, 255))
        y += 6
        line(f"TRACKED OBJECTS ({len(objs)})  [array in memory]", (255, 255, 0))
        for o in pend[:10]:
            line(f" {order[o.id]}. #{o.id} {CLASS_NAMES[o.cls]:<11} ({o.x:4.0f},{o.y:4.0f}) {(o.hit_time - now) * 1000:+5.0f}ms")
        y += 6
        line("EVENTS", (255, 255, 0))
        for e in events:
            line(e, (190, 190, 190))
        return np.hstack([view, panel])

    def run(self):
        cv2.namedWindow("osu-ai", cv2.WINDOW_NORMAL)
        while not self.shared.quit:
            cv2.imshow("osu-ai", self.render())
            k = cv2.waitKey(15) & 0xFF
            if k == ord("q"):
                self.shared.quit = True
            elif k == ord(" "):
                self.shared.armed = not self.shared.armed
                self.shared.log("ARMED" if self.shared.armed else "DISARMED")
        cv2.destroyAllWindows()
