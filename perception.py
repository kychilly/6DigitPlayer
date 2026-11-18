from __future__ import annotations
import os
import threading
import time
import cv2
import numpy as np
import torch
from config import ModelConfig
from contextlib import nullcontext
from model import OsuNet, decode, pick_device, device_name


class Detector:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.device = pick_device(cfg.device)
        print(f"[detector] device: {self.device} ({device_name(self.device)})")
        self.model = OsuNet(cfg.num_classes).to(self.device).eval()
        if os.path.exists(cfg.weights):
            ck = torch.load(cfg.weights, map_location=self.device)
            self.model.load_state_dict(ck["model"] if "model" in ck else ck)
            print(f"[detector] loaded {cfg.weights} on {self.device}")
        else:
            print(f"[detector] WARNING: {cfg.weights} not found -> random weights. Run train.py first.")
        torch.backends.cudnn.benchmark = True

    @torch.no_grad()
    def infer(self, frame_bgr: np.ndarray):
        c = self.cfg
        small = cv2.resize(frame_bgr, (c.input_w, c.input_h), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray(small[:, :, ::-1])).permute(2, 0, 1).float().div_(255.0)
        x = x.unsqueeze(0).to(self.device, non_blocking=True)
        ctx = torch.autocast(device_type="cuda") if self.device.type == "cuda" else nullcontext()
        with ctx:
            heat, reg = self.model(x)
        return decode(heat, reg, c.stride, c.score_threshold, c.max_dets)[0]


class PerceptionThread(threading.Thread):
    """capture frame -> CNN -> detections -> tracker (the bot's only source of map info)."""

    def __init__(self, capture, detector: Detector, tracker, shared, mcfg: ModelConfig, capcfg):
        super().__init__(daemon=True)
        self.capture, self.detector, self.tracker, self.shared = capture, detector, tracker, shared
        self.scale = capcfg.width / mcfg.input_w
        self.frame_h = capcfg.height
        self._stop_flag = False

    def run(self):
        last_seq, n, t_fps = -1, 0, time.perf_counter()
        while not self._stop_flag:
            seq, ts, frame = self.capture.latest()
            if frame is None or seq == last_seq:
                time.sleep(0.001)
                continue
            last_seq = seq
            t0 = time.perf_counter()
            dets = self.detector.infer(frame)
            infer_ms = (time.perf_counter() - t0) * 1000
            self.tracker.update(dets, ts, self.scale, self.frame_h)
            self.shared.set_frame(frame, ts)
            n += 1
            now = time.perf_counter()
            with self.shared.lock:
                self.shared.stats["infer_ms"] = infer_ms
                self.shared.stats["capture_fps"] = self.capture.fps
                if now - t_fps > 1.0:
                    self.shared.stats["perception_fps"] = n / (now - t_fps)
                    n, t_fps = 0, now

    def stop(self):
        self._stop_flag = True
