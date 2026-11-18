"""Mouse/keyboard output. Windows uses SendInput (absolute moves + scan-code keys).
Everywhere else (or --dry-run) a NullInput is used so you can develop safely."""
from __future__ import annotations
import os
import platform


class NullInput:
    def __init__(self, center=(960, 540)):
        self._pos = center

    def get_pos(self):
        return self._pos

    def move(self, x, y):
        self._pos = (x, y)

    def key_down(self, key):
        pass

    def key_up(self, key):
        pass


if platform.system() == "Windows":
    import ctypes
    from ctypes import wintypes  # noqa: F401

    PUL = ctypes.POINTER(ctypes.c_ulong)

    class _KeyBd(ctypes.Structure):
        _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort), ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]

    class _Mouse(ctypes.Structure):
        _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_ulong),
                    ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]

    class _Hw(ctypes.Structure):
        _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_short), ("wParamH", ctypes.c_ushort)]

    class _II(ctypes.Union):
        _fields_ = [("ki", _KeyBd), ("mi", _Mouse), ("hi", _Hw)]

    class _Input(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("ii", _II)]

    SCAN = {"[": 0x1A, "]": 0x1B, "z": 0x2C, "x": 0x2D, "c": 0x2E, "v": 0x2F, "a": 0x1E, "s": 0x1F}
    MOUSE_BTN = {"mouse1": (0x0002, 0x0004), "mouse2": (0x0008, 0x0010)}

    class WindowsInput:
        def __init__(self):
            u = ctypes.windll.user32
            try:
                u.SetProcessDPIAware()
            except Exception:
                pass
            self.u = u
            self.vx, self.vy = u.GetSystemMetrics(76), u.GetSystemMetrics(77)
            self.vw, self.vh = u.GetSystemMetrics(78), u.GetSystemMetrics(79)
            self._last = None
            self._extra = ctypes.c_ulong(0)

        def _send(self, inp):
            self.u.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

        def get_pos(self):
            pt = ctypes.wintypes.POINT()
            self.u.GetCursorPos(ctypes.byref(pt))
            return (pt.x, pt.y)

        def move(self, x, y):
            ix, iy = int(round(x)), int(round(y))
            if self._last == (ix, iy):
                return
            self._last = (ix, iy)
            nx = int((ix - self.vx) * 65535 / max(1, self.vw - 1))
            ny = int((iy - self.vy) * 65535 / max(1, self.vh - 1))
            mi = _Mouse(nx, ny, 0, 0x0001 | 0x8000 | 0x4000, 0, ctypes.pointer(self._extra))
            self._send(_Input(0, _II(mi=mi)))

        def _key(self, key, up):
            if key in MOUSE_BTN:
                d, u_ = MOUSE_BTN[key]
                mi = _Mouse(0, 0, 0, u_ if up else d, 0, ctypes.pointer(self._extra))
                self._send(_Input(0, _II(mi=mi)))
                return
            if key not in SCAN:
                raise ValueError(f"Unsupported key {key!r}; add its scan code to SCAN in inputs.py")
            flags = 0x0008 | (0x0002 if up else 0)
            ki = _KeyBd(0, SCAN[key], flags, 0, ctypes.pointer(self._extra))
            self._send(_Input(1, _II(ki=ki)))

        def key_down(self, key):
            self._key(key, False)

        def key_up(self, key):
            self._key(key, True)


def foreground_info():
    """(exe name, window title) of the focused window; (None, None) off-Windows."""
    if platform.system() != "Windows":
        return None, None
    try:
        u, k = ctypes.windll.user32, ctypes.windll.kernel32
        u.GetForegroundWindow.restype = ctypes.c_void_p
        u.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        u.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
        k.OpenProcess.restype = ctypes.c_void_p
        k.CloseHandle.argtypes = [ctypes.c_void_p]
        k.QueryFullProcessImageNameW.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_wchar_p,
                                                 ctypes.POINTER(ctypes.c_ulong)]
        hwnd = u.GetForegroundWindow()
        if not hwnd:
            return None, None
        tbuf = ctypes.create_unicode_buffer(512)
        u.GetWindowTextW(hwnd, tbuf, 512)
        pid = ctypes.c_ulong(0)
        u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        h = k.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return None, tbuf.value
        buf, size = ctypes.create_unicode_buffer(520), ctypes.c_ulong(520)
        ok = k.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size))
        k.CloseHandle(h)
        return (os.path.basename(buf.value).lower() if ok else None), tbuf.value
    except Exception:
        return None, None


def check_foreground(prefix: str = "osu", title_gate: bool = False):
    """-> (ok, process, title). ok = osu! is focused (and, with title_gate, its title says a map is playing)."""
    if platform.system() != "Windows":
        return True, "", ""
    proc, title = foreground_info()
    ok = bool(proc) and proc.startswith(prefix.lower())
    if ok and title_gate:
        t = (title or "").strip().lower()
        ok = t.startswith("osu!") and " - " in t
    return ok, proc or "", title or ""


def enable_high_res_timer():
    if platform.system() == "Windows":
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass


def make_backend(dry_run: bool, center=(960, 540)):
    if dry_run or platform.system() != "Windows":
        if not dry_run:
            print("[input] Non-Windows platform: using NullInput (no real mouse/keyboard output).")
        return NullInput(center)
    return WindowsInput()
