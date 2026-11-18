"""Listens to the game audio (system loopback), detects note onsets with spectral flux,
estimates tempo from inter-onset intervals, and lets the tap planner snap its visually
estimated hit time to the music's rhythm."""
from __future__ import annotations
import threading
import time
from collections import deque
import numpy as np
from config import AudioConfig


class AudioListener(threading.Thread):
    def __init__(self, cfg: AudioConfig):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.lock = threading.Lock()
        self.onsets: deque = deque(maxlen=256)   # (time, strength)
        self.period: float | None = None
        self.confidence = 0.0
        self.available = False
        self._stop_flag = False
        self._hist: deque = deque(maxlen=int(1.5 * cfg.samplerate / cfg.hop))

    # ----------------------------------------------------------------- thread
    def run(self):
        c = self.cfg
        try:
            import soundcard as sc
            spk = sc.default_speaker()
            mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
        except Exception as e:  # noqa
            print(f"[audio] disabled ({e}). pip install soundcard for beat listening.")
            return
        win = np.hanning(c.n_fft).astype(np.float32)
        buf = np.zeros(c.n_fft, np.float32)
        freqs = np.fft.rfftfreq(c.n_fft, 1.0 / c.samplerate)
        band = (freqs > 60) & (freqs < 9000)
        prev = None
        with mic.recorder(samplerate=c.samplerate, channels=1, blocksize=c.hop) as rec:
            self.available = True
            while not self._stop_flag:
                x = rec.record(numframes=c.hop)[:, 0].astype(np.float32)
                t_end = time.perf_counter()
                buf = np.roll(buf, -len(x))
                buf[-len(x):] = x
                spec = np.log1p(20 * np.abs(np.fft.rfft(buf * win)))[band]
                flux = float(np.maximum(spec - prev, 0).sum()) if prev is not None else 0.0
                prev = spec
                self._pick(flux, t_end)

    def stop(self):
        self._stop_flag = True

    # ----------------------------------------------------------------- analysis
    def _pick(self, flux: float, t_end: float):
        h = self._hist
        h.append((flux, t_end))
        if len(h) < 12:
            return
        f = np.array([v[0] for v in h])
        thr = f.mean() + 1.3 * f.std() + 1e-3
        a, b, c = f[-3], f[-2], f[-1]
        tb = h[-2][1]
        with self.lock:
            last = self.onsets[-1][0] if self.onsets else -1
            if b > thr and b >= a and b > c and tb - last > 0.06:
                t_on = tb - 0.5 * self.cfg.hop / self.cfg.samplerate + self.cfg.offset
                self.onsets.append((t_on, float(b / thr)))
                self._update_tempo()

    def _update_tempo(self):
        now = self.onsets[-1][0]
        ts = np.array([t for t, _ in self.onsets if now - t < 6.0])
        if len(ts) < 6:
            return
        iois = []
        for k in (1, 2):
            iois.extend(np.diff(ts, n=1)[:] if k == 1 else (ts[2:] - ts[:-2]))
        iois = np.array([v for v in iois if 0.15 <= v <= 1.0])
        if len(iois) < 5:
            return
        hist, edges = np.histogram(iois, bins=np.arange(0.15, 1.0, 0.01))
        i = int(hist.argmax())
        self.period = float(edges[i] + 0.005)
        self.confidence = float(hist[i] / len(iois))

    # ----------------------------------------------------------------- API
    def refine(self, t_est: float, window: float = 0.04, weight: float = 0.5) -> float:
        """Pull a visually estimated hit time toward the nearest audio onset
        (or predicted beat on the tempo grid) if one is within `window`."""
        with self.lock:
            onsets = [o[0] for o in self.onsets]
            period, conf = self.period, self.confidence
        if not onsets:
            return t_est
        cand = min(onsets, key=lambda o: abs(o - t_est))
        if abs(cand - t_est) <= window:
            return t_est + weight * (cand - t_est)
        if period and conf > 0.3 and t_est > onsets[-1]:
            k = round((t_est - onsets[-1]) / period)
            grid = onsets[-1] + k * period
            if k >= 1 and abs(grid - t_est) <= window:
                return t_est + 0.5 * weight * (grid - t_est)
        return t_est

    def info(self) -> dict:
        with self.lock:
            last = self.onsets[-1][0] if self.onsets else None
            return {"available": self.available,
                    "bpm": 60.0 / self.period if self.period else None,
                    "conf": self.confidence,
                    "since_onset": (time.perf_counter() - last) if last else None}
