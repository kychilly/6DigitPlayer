"""Global emergency stop + arm toggle. Polls the physical keyboard through Windows,
so it works no matter which window has focus or what the bot is doing.
  ` (backtick, the key left of 1) : release keys and kill the whole process immediately
  F8                              : arm / disarm"""
from __future__ import annotations
import os
import platform
import sys
import threading
import time

VK_BACKTICK, VK_F8 = 0xC0, 0x77   # VK_OEM_3 = ` / ~ on US keyboards

class KillSwitch(threading.Thread):
    def __init__(self, shared, backend, keys):
        super().__init__(daemon=True)
        self.shared, self.backend = shared, backend
        self.keys = tuple(keys) + ("mouse1", "mouse2")

    def _release_everything(self):
        for k in self.keys:
            try:
                self.backend.key_up(k)
            except Exception:
                pass

    def _kill(self):
        self.shared.armed = False          # controller stops sending immediately
        self.shared.quit = True
        self._release_everything()
        time.sleep(0.005)                  # cover a key-down that was mid-flight
        self._release_everything()
        print("\n[KILL] ` (backtick) pressed - bot shut down.", flush=True)
        sys.stdout.flush()
        os._exit(0)                        # hard exit: nothing can keep running

    def run(self):
        if platform.system() != "Windows":
            print("[killswitch] Windows only; use Ctrl+C in the terminal.")
            return
        import ctypes
        u = ctypes.windll.user32
        prev_f8 = False
        while True:
            if u.GetAsyncKeyState(VK_BACKTICK) & 0x8000:
                self._kill()
            f8 = bool(u.GetAsyncKeyState(VK_F8) & 0x8000)
            if f8 and not prev_f8:
                self.shared.armed = not self.shared.armed
                self.shared.log("ARMED" if self.shared.armed else "DISARMED")
            prev_f8 = f8
            time.sleep(0.002)
