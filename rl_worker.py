"""TRUE reinforcement learning after every map.

run_bot.py --policy starts this automatically (as a separate, low-priority process). You can also run it by hand:

    python rl_worker.py --watch                     # what run_bot does: learn after every finished map
    python rl_worker.py --once                      # score + learn from any episodes still waiting, then exit
    python rl_worker.py --episode data/rl_episodes/X.npz --replay "path/to/play.osr"
    python rl_worker.py --rollback                  # undo the last update
    python rl_worker.py --stats                     # print the learning log (is it improving?)
    python rl_worker.py --study                     # study the replay library now (no osu! needed), then exit
    python rl_worker.py --fresh                     # throw away the current policy, start a NEW one from the library

How one map is learned
  1. While playing, the bot records every decision it made (what it saw, what it did, how likely that action was,
     its value estimate, and when each key went down) -> data/rl_episodes/<time>.npz
  2. osu! saves the replay of the play. The worker finds the replay that belongs to the episode by lining up the
     bot's key presses with the replay's key presses (this also measures the clock offset between the two).
  3. Every step of the episode is re-scored with osu!'s REAL rules on the real beatmap (judge.py, the same reward
     the simulator uses): 300 = +1.0, 100 = +0.5, 50 = +0.2, miss = 0, stray click = -0.05, a bonus for tapping
     near 0 ms, plus shaping for moving toward the next note, following slider balls and spinning.
     If no replay turns up (failed/quit map, osu!lazer) it falls back to the bot's own live judgement.
  4. PPO (ppo.py) updates the policy from that episode: actions that led to more reward than the critic expected
     become more likely, actions that led to less become less likely. Each update is size-limited (PPO clip +
     KL early stop) so one bad map can't wreck the policy.
  5. REPLAY LIBRARY (replay_library/, max 300 replays): your own good plays (.osr files you drop in replay_library/)
     plus the bot's own past plays (copied automatically to replay_library/bot_plays/). Every replay is scored with
     the same reward, and the policy is trained on ALL of them with advantage-weighted regression (offline RL):
     moments that scored better than the critic expected are copied strongly, moments that scored worse are mostly
     ignored. So it learns "what a 300 looks like" from your plays and from its own best moments.
     When the worker starts and the library has changed, it first does a longer study session, so the very first
     map is already played with what it learned. After every map it does a shorter refresher.
  6. The new weights are written atomically; the running bot hot-reloads them before the next map.
     Every map's diagnosis (hits, timing offset, aim misses, early clicks, slider follow...) goes to data/rl_log.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from config import BotConfig, ar_to_preempt
from judge import Judge, RewardShaper, hit_windows, spinner_required_revs
from obs import VisObj, CIRCLE as V_CIRCLE, SLIDER_HEAD as V_SLIDER_HEAD, SPINNER as V_SPINNER, sort_visible
from osu_formats import parse_osr, parse_osu, BAD_MODS, MOD_HR, MOD_EZ, SLIDER, SPINNER
from policy import load_policy, save_policy

LOCAL = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "osu!"
DEFAULT_SONGS = str(LOCAL / "Songs")
DEFAULT_REPLAY_DIRS = [str(LOCAL / "Replays"), str(LOCAL / "Data" / "r")]
EP_DIR = Path("data/rl_episodes")
DONE_DIR = EP_DIR / "done"
STATE_PATH = Path("checkpoints/rl_state.pt")
USED_PATH = Path("data/rl_used_replays.json")
LOG_PATH = Path("data/rl_log.csv")
HIST_DIR = Path("checkpoints/history")
LIB_DIR = Path("replay_library")              # drop your own good .osr files here
BOT_LIB = LIB_DIR / "bot_plays"               # the bot's own plays are copied here automatically
CACHE_DIR = Path("data/replay_cache")         # each replay scored once, then cached

PRESS_TOL = 0.025          # s: a bot key press and a replay key press closer than this are "the same press"


# ====================================================================== replay helpers
def replay_presses(rep) -> np.ndarray:
    """(N,3) [time, x, y] of every key-down in the replay (both keys in the same frame = one click)."""
    out = []
    for ki in (0, 1):
        s = rep.key_state(ki)
        prev = np.concatenate([[False], s[:-1]])
        for k in np.nonzero(s & ~prev)[0]:
            out.append((rep.t[k], rep.x[k], rep.y[k]))
    out.sort()
    res, last = [], -1e9
    for p in out:
        if p[0] - last >= 1e-4:
            res.append(p)
            last = p[0]
    return np.array(res, float).reshape(-1, 3)


def align(live_t: np.ndarray, rep_t: np.ndarray, tol: float = PRESS_TOL):
    """Find c so that live_press_time + c == replay_press_time. -> (c, matched, frac, resid_ms) or None."""
    live_t, rep_t = np.sort(np.asarray(live_t, float)), np.sort(np.asarray(rep_t, float))
    if len(live_t) < 4 or len(rep_t) < 4:
        return None

    def score(c):
        s = live_t + c
        j = np.clip(np.searchsorted(rep_t, s), 1, len(rep_t) - 1)
        d = np.minimum(np.abs(rep_t[j] - s), np.abs(rep_t[j - 1] - s))
        return d

    picks = np.unique(np.concatenate([np.arange(min(12, len(live_t))),
                                      np.linspace(0, len(live_t) - 1, min(12, len(live_t))).astype(int)]))
    best_c, best_n = None, -1
    for i in picks:
        for c in rep_t - live_t[i]:
            n = int((score(c) < tol).sum())
            if n > best_n:
                best_c, best_n = c, n
    for _ in range(3):                                         # refine with the median residual of matched presses
        s = live_t + best_c
        j = np.clip(np.searchsorted(rep_t, s), 1, len(rep_t) - 1)
        near = np.where(np.abs(rep_t[j] - s) < np.abs(rep_t[j - 1] - s), rep_t[j], rep_t[j - 1])
        res = near - s
        m = np.abs(res) < tol
        if m.sum() == 0:
            break
        best_c += float(np.median(res[m]))
    d = score(best_c)
    m = d < tol
    return best_c, int(m.sum()), float(m.mean()), float(d[m].std() * 1000) if m.any() else 0.0


# ====================================================================== scoring an episode with the real rules
def score_episode(bm, rep, t_map: np.ndarray, cfg: BotConfig):
    """Re-play the replay against the real beatmap on the episode's own step times.
    -> (per-step reward (T,), diagnosis dict). Uses exactly the simulator's reward (judge.RewardShaper)."""
    rc = cfg.reward
    T = len(t_map)
    judge = Judge(bm.objects, bm.od, bm.radius)
    shaper = RewardShaper(rc, bm.od)
    w300, w100, w50 = hit_windows(bm.od)
    preempt = ar_to_preempt(bm.ar)
    objs = bm.objects
    presses = replay_presses(rep)

    px, py = np.interp(t_map, rep.t, rep.x), np.interp(t_map, rep.t, rep.y)
    idx = np.clip(np.searchsorted(rep.t, t_map, side="right") - 1, 0, len(rep.t) - 1)
    outside = (t_map < rep.t[0]) | (t_map > rep.t[-1])
    k0 = rep.key_state(0)[idx] & ~outside
    k1 = rep.key_state(1)[idx] & ~outside

    diag = {"too_early": 0, "aim_miss": 0, "no_note": 0, "hit_dts": []}
    r = np.zeros(T, np.float32)
    hit_r = np.zeros(T, np.float32)                  # the part of r that comes from 300/100/50 judgements
    pi, ptr, live = 0, 0, []
    prev_t = t_map[0] - 1.0 / cfg.policy.hz
    for k in range(T):
        t = float(t_map[k])
        events = []
        while pi < len(presses) and presses[pi][0] <= t:
            pt, x, y = presses[pi]
            pi += 1
            pre = judge.advance(pt)
            if judge.i >= len(judge.q):
                kind = "no_note"
            else:
                o = judge.q[judge.i]
                dt = pt - o.time
                kind = ("too_early" if dt < -w50 else
                        None if math.hypot(x - o.x, y - o.y) <= judge.r else "aim_miss")
            ev = judge.press(pt, (x, y))
            if pt > prev_t - 1e-9:                             # presses before the episode started don't count
                events += pre + ev
                if kind:
                    diag[kind] += 1
                for e in ev:
                    if e.kind in ("300", "100", "50", "head") and e.obj >= 0 and e.dt != 0.0:
                        diag["hit_dts"].append(e.dt)
        events += judge.advance(t)
        dt_k = max(t - prev_t, 1e-4)
        follow_ok, revs, spin = judge.track(t, dt_k, (px[k], py[k]), bool(k0[k] or k1[k]))
        while ptr < len(objs) and objs[ptr].time - preempt <= t:
            live.append(objs[ptr]); ptr += 1
        keep, vis = [], []
        for o in live:
            if o.kind == SPINNER:
                if t > o.end_time:
                    continue
                kind = V_SPINNER
            else:
                if o.index in judge.resolved:
                    continue
                kind = V_SLIDER_HEAD if o.kind == SLIDER else V_CIRCLE
            keep.append(o)
            vis.append(VisObj(o.index, kind, o.x, o.y, bm.radius, o.time - t))
        live = keep
        need = spinner_required_revs(spin.end_time - spin.time, bm.od) if spin is not None else 0.0
        r[k] = shaper.step(events, (px[k], py[k]), sort_visible(vis), (bool(k0[k]), bool(k1[k])), follow_ok,
                           revs, spin.index if spin is not None else None, need)
        hit_r[k] = hit_reward(events, rc)
        prev_t = t
    judge.finalize(1e9)

    c = judge.counts
    dts = np.array(diag.pop("hit_dts")) * 1000.0
    fol = [v for v in judge.follow.values() if v[1] > 0]
    spins = [(judge.revs[o.index], spinner_required_revs(o.end_time - o.time, bm.od)) for o in objs if o.kind == SPINNER]
    diag.update(counts=dict(c), acc=judge.accuracy(),
                offset_mean_ms=float(dts.mean()) if len(dts) else 0.0,
                offset_std_ms=float(dts.std()) if len(dts) else 0.0,
                slider_follow=float(sum(v[0] for v in fol) / max(sum(v[1] for v in fol), 1e-9)) if fol else None,
                spinner_done=float(np.mean([min(a / b, 1.0) for a, b in spins])) if spins else None,
                w300_ms=w300 * 1000)

    # ---- osu!'s own score is the authority: scale the judgement rewards so this play's total matches the
    #      300/100/50 counts osu! itself saved in the replay (the reconstruction only decides WHICH moments get it)
    official = (int(rep.n300), int(rep.n100), int(rep.n50), int(rep.nmiss))
    if sum(official) > 0:
        pts = lambda a, b, cc: 300 * a + 100 * b + 50 * cc
        mine = pts(c["300"], c["100"], c["50"])
        ratio = float(np.clip(pts(*official[:3]) / mine, 0.5, 1.5)) if mine > 0 else 1.0
        r = r + hit_r * (ratio - 1.0)
        n = sum(official)
        diag.update(official=dict(zip(("300", "100", "50", "miss"), official)),
                    official_acc=pts(*official[:3]) / (300.0 * n), reconstructed_acc=diag["acc"], scale=ratio)
        diag["acc"] = diag["official_acc"]
    return r, diag


def hit_reward(events, rc) -> float:
    """Reward from judgement events only (same numbers as RewardShaper, without the shaping)."""
    r = 0.0
    for e in events:
        if e.kind in ("300", "100", "50", "head"):
            r += {"300": rc.r300, "100": rc.r100, "50": rc.r50, "head": rc.slider_head}[e.kind]
            if e.dt != 0.0:
                r += rc.timing_bonus * math.exp(-(e.dt / rc.timing_sigma) ** 2)
    return r


def describe(diag: dict) -> list:
    """Human-readable 'what went well / what went badly' lines."""
    c, out = diag.get("official") or diag["counts"], []
    n = c["300"] + c["100"] + c["50"] + c["miss"]
    src = "osu! score" if "official" in diag else "acc"
    out.append(f"{src}: {diag['acc'] * 100:.1f}%  300 {c['300']} | 100 {c['100']} | 50 {c['50']} | miss {c['miss']}  "
               f"({n} notes)")
    if "official" in diag:
        out.append(f"(per-note replay reconstruction: {diag['reconstructed_acc'] * 100:.1f}% - used only to decide "
                   f"which moments earned the reward)")
    m, s, w = diag["offset_mean_ms"], diag["offset_std_ms"], diag["w300_ms"]
    if c["300"] + c["100"] + c["50"]:
        lean = "late" if m > 3 else "early" if m < -3 else "centred"
        out.append(f"timing: {m:+.1f} ms on average ({lean}), spread +/-{s:.1f} ms (300 window is +/-{w:.0f} ms)")
    bad = []
    if diag["aim_miss"]:
        bad.append(f"{diag['aim_miss']} clicks on time but off the circle (aim)")
    if diag["too_early"]:
        bad.append(f"{diag['too_early']} clicks far too early")
    if diag["no_note"]:
        bad.append(f"{diag['no_note']} clicks with nothing to hit")
    if bad:
        out.append("mistakes: " + "; ".join(bad))
    if diag.get("slider_follow") is not None:
        out.append(f"slider follow: {diag['slider_follow'] * 100:.0f}% of slider time on the ball with a key held")
    if diag.get("spinner_done") is not None:
        out.append(f"spinners: {diag['spinner_done'] * 100:.0f}% of required spins")
    return out


# ====================================================================== replay library + offline RL (AWR)
def file_md5(p) -> str:
    import hashlib
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def reward_signature(cfg: BotConfig) -> str:
    """Changes whenever the reward settings change -> cached replay scores are recomputed."""
    import hashlib
    from judge import JUDGE_VERSION
    return hashlib.md5((repr(sorted(vars(cfg.reward).items())) + JUDGE_VERSION).encode()).hexdigest()[:12]


def replay_to_samples(bm, rep, cfg: BotConfig):
    """A replay (anyone's) -> (obs, act, reward) on the policy's 120 Hz grid, scored with the real rules."""
    from replay_data import judge_replay, build_samples
    judge = judge_replay(bm, rep)
    obs, act = build_samples(bm, rep, judge, cfg)
    dt = 1.0 / cfg.policy.hz
    t0 = bm.objects[0].time - ar_to_preempt(bm.ar) - 0.3
    t1 = max(o.end_time for o in bm.objects) + 0.5
    grid = np.arange(t0, t1, dt)
    t_next = grid[1:len(obs) + 1]                      # reward for action k = what happened by the next step
    rew, diag = score_episode(bm, rep, t_next, cfg)
    end = int(np.searchsorted(t_next, rep.t[-1] + 0.5))   # quit / failed early: drop the empty tail
    end = max(end, min(len(obs), 120))
    return obs[:end], act[:end], rew[:end], diag


LIB_STRIDE = 2      # library replays are studied at 60 Hz (every 2nd step): half the memory, same information


def downsample(obs, act, rew, k: int = LIB_STRIDE):
    """Every k-th step (float16 to save memory); rewards of the skipped steps are summed into the kept one."""
    n = len(rew) // k * k
    if n < k:
        return None
    return (np.ascontiguousarray(obs[:n:k]).astype(np.float16), np.ascontiguousarray(act[:n:k]).astype(np.float16),
            rew[:n].reshape(-1, k).sum(1).astype(np.float32))


def gae_np(rew, val, gamma, lam):
    T = len(rew)
    adv = np.zeros(T, np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nv = val[t + 1] if t + 1 < T else 0.0
        nonterm = 1.0 if t + 1 < T else 0.0
        delta = rew[t] + gamma * nv * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + val


def awr_train(model, episodes, cfg: BotConfig, dev, steps: int, lr: float, beta: float = 1.0,
              critic_steps: int = 0, batch: int = 1024, log=print):
    """Advantage-weighted regression (offline RL) over a list of (obs, act, rew) episodes.
    critic learns V(s) from the real rewards; actor imitates each action with weight exp(advantage / beta),
    so moves that turned out better than expected are learned strongly and bad ones are mostly ignored."""
    import torch.nn.functional as Fn
    from policy import augment
    if not episodes or steps <= 0:
        return None
    gamma, lam = cfg.reward.gamma ** LIB_STRIDE, 0.95 ** LIB_STRIDE     # episodes are downsampled
    obs = torch.from_numpy(np.concatenate([e[0] for e in episodes])).to(dev)      # float16
    act = torch.from_numpy(np.concatenate([e[1] for e in episodes])).to(dev)
    N = len(obs)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def targets():
        advs, rets = [], []
        with torch.no_grad():
            for o, _, r in episodes:
                v = model(torch.from_numpy(o).to(dev).float())[3].cpu().numpy()
                a_, r_ = gae_np(r.astype(np.float32), v.astype(np.float32), gamma, lam)
                advs.append(a_); rets.append(r_)
        adv = np.concatenate(advs)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-6)
        w = np.minimum(np.exp(adv_n / beta), 20.0).astype(np.float32)
        return torch.from_numpy(w).to(dev), torch.from_numpy(np.concatenate(rets).astype(np.float32)).to(dev)

    w, ret = targets()
    model.train()
    st = {"actor": 0.0, "vf": 0.0, "n": 0}
    for s in range(steps):
        if s and s % 250 == 0:
            w, ret = targets()                          # critic improved -> re-judge which moves were good
        i = torch.randint(0, N, (min(batch, N),), device=dev)
        o, a = augment(obs[i].float(), act[i].float(), cfg.policy)
        mu, logstd, kl, v = model(o)
        vf = 0.5 * (v - ret[i]).pow(2).mean()
        loss = 0.5 * vf
        if s >= critic_steps:
            std = logstd.exp()
            nll = (0.5 * ((a[:, :2] - mu) / std) ** 2 + logstd).sum(-1)
            ce = Fn.binary_cross_entropy_with_logits(kl, a[:, 2:4], reduction="none").sum(-1)
            wi = w[i]
            actor = (wi * (nll + ce)).sum() / wi.sum()
            loss = loss + actor
            st["actor"] += float(actor.detach()); st["n"] += 1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        st["vf"] += float(vf.detach())
        if log and steps >= 1000 and (s + 1) % 500 == 0:
            log(f"  study {s + 1}/{steps}  value err {st['vf'] / (s + 1):.3f}")
    model.eval()
    return {"actor": st["actor"] / max(st["n"], 1), "vf": st["vf"] / steps, "samples": N}


# ====================================================================== the worker
class Worker:
    def __init__(self, a):
        self.a = a
        self.cfg = BotConfig()
        self.dev = torch.device(a.device)
        self.idx = None
        self.used = set(json.loads(USED_PATH.read_text())) if USED_PATH.exists() else set()
        self.human = None
        self._lib_mem: dict = {}
        self._lib_acc: dict = {}
        if a.human_data and os.path.exists(a.human_data):
            d = np.load(a.human_data)
            if int(d["n_objs"]) == self.cfg.policy.n_objs:
                self.human = a.human_data
            else:
                print(f"[rl] {a.human_data} was built for a different n_objs; not mixing it in")

    def log(self, msg):
        print(f"[rl] {msg}", flush=True)

    # ---------------------------------------------------------------- finding the replay of an episode
    def _beatmap_for(self, rep):
        from replay_data import index_songs
        def build():
            idx = {}
            for d in [self.a.songs] + list(getattr(self.a, "extra_songs", []) or []):
                if d and os.path.isdir(d):
                    idx.update(index_songs(d))
            return idx
        if self.idx is None:
            self.idx = build()
        osu = self.idx.get(rep.beatmap_md5)
        if osu is None:
            self.idx = build()                            # new map downloaded since the last index
            osu = self.idx.get(rep.beatmap_md5)
        return osu

    def candidate_replays(self, wall_end: float):
        out = []
        for d in self.a.replay_dirs:
            if not os.path.isdir(d):
                continue
            for p in Path(d).glob("*.osr"):
                sp = str(p)
                if sp in self.used:
                    continue
                try:
                    m = p.stat().st_mtime
                except OSError:
                    continue
                if wall_end - 1800 <= m <= wall_end + 600 and time.time() - m > 1.0:   # fully written
                    out.append(sp)
        return out

    def match(self, ep, replays):
        live_t = ep["t"][ep["press"].any(1)]
        best = None
        for p in replays:
            try:
                rep = parse_osr(p)
            except Exception:                              # noqa  (not a replay / still being written)
                continue
            if rep.mode != 0 or len(rep.t) < 10:
                continue
            res = align(live_t, replay_presses(rep)[:, 0])
            if res is None:
                continue
            c, n, frac, resid = res
            if n >= max(5, 0.6 * len(live_t)) and (best is None or n > best[2]):
                best = (p, rep, n, c, frac, resid)
        return best

    # ---------------------------------------------------------------- learning
    def rewards_for(self, ep, replay_path=None):
        """-> (rewards, source, diag, info)"""
        replays = [replay_path] if replay_path else self.candidate_replays(float(ep["wall_end"]))
        m = self.match(ep, replays) if replays else None
        if m is None:
            return None
        p, rep, n, c, frac, resid = m
        if rep.mods & BAD_MODS:
            self.log(f"replay {Path(p).name} uses speed/auto mods ({rep.mods}); using the live judgement instead")
            self.used.add(p)
            return "bad_mods"
        osu = self._beatmap_for(rep)
        if osu is None:
            self.log(f"replay {Path(p).name}: its beatmap isn't in {self.a.songs}; using the live judgement instead")
            self.used.add(p)
            return "no_map"
        bm = parse_osu(osu).with_mods(rep.mods)
        t_map = ep["t"] + c
        r, diag = score_episode(bm, rep, t_map, self.cfg)
        self.used.add(p)
        self.add_bot_play(p, bm.title)
        info = {"replay": Path(p).name, "map": f"{bm.title}", "ar": bm.ar, "od": bm.od, "cs": bm.cs,
                "matched": f"{n}/{int(ep['press'].any(1).sum())} presses (+/-{resid:.1f} ms)"}
        told_ar, told_od = float(ep["ar"]), float(ep["od"])
        if abs(told_ar - bm.ar) > 0.25 or abs(told_od - bm.od) > 0.25:
            info["warning"] = (f"the bot played with AR {told_ar:.1f} / OD {told_od:.1f} but the map is "
                               f"AR {bm.ar:.1f} / OD {bm.od:.1f}{' (HR)' if rep.mods & MOD_HR else ''}"
                               f"{' (EZ)' if rep.mods & MOD_EZ else ''} - its note timing was off this map")
        return r, "replay", diag, info

    # ---------------------------------------------------------------- replay library
    def library_files(self) -> list:
        """Your replays first (newest first), then the bot's own plays, max --max-replays in total."""
        cap = self.a.max_replays
        mine = sorted(LIB_DIR.glob("*.osr"), key=lambda p: -p.stat().st_mtime) if LIB_DIR.is_dir() else []
        bot = sorted(BOT_LIB.glob("*.osr"), key=lambda p: -p.stat().st_mtime) if BOT_LIB.is_dir() else []
        files = mine[:cap] + bot[:max(0, cap - len(mine))]
        for p in bot[max(0, cap - len(mine)):]:           # keep the folder from growing forever
            try:
                p.unlink()
            except OSError:
                pass
        return files

    def library_episode(self, p: Path):
        """-> (obs, act, rew) for one library replay (cached), or None if unusable."""
        try:
            h = file_md5(p)
        except OSError:
            return None
        if h in self._lib_mem:
            return self._lib_mem[h]
        cp = CACHE_DIR / f"{h}.npz"
        if cp.exists():
            d = np.load(cp)
            if (int(d["n_objs"]) == self.cfg.policy.n_objs and int(d["hz"]) == self.cfg.policy.hz
                    and "rsig" in d.files and str(d["rsig"]) == reward_signature(self.cfg)):
                self._lib_acc[h] = float(d["acc"])
                ep = downsample(d["obs"], d["act"], d["rew"])
                self._lib_mem[h] = ep
                return ep
        try:
            rep = parse_osr(p)
        except Exception as e:  # noqa
            self.log(f"library: can't read {p.name} ({e}) - skipped"); self._lib_mem[h] = None; return None
        if rep.mode != 0 or rep.mods & BAD_MODS:
            self.log(f"library: {p.name} is not plain osu!standard (mods {rep.mods}) - skipped")
            self._lib_mem[h] = None; return None
        osu = self._beatmap_for(rep)
        if osu is None:
            self.log(f"library: the map for {p.name} isn't installed - skipped"); self._lib_mem[h] = None; return None
        bm = parse_osu(osu).with_mods(rep.mods)
        obs, act, rew, diag = replay_to_samples(bm, rep, self.cfg)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cp, obs=obs, act=act, rew=rew, acc=np.float32(diag["acc"]),
                            n_objs=self.cfg.policy.n_objs, hz=self.cfg.policy.hz,
                            rsig=np.array(reward_signature(self.cfg)))
        c = diag["counts"]
        self.log(f"library: + {p.name[:60]}  ({diag['acc'] * 100:.1f}%  {c['300']}/{c['100']}/{c['50']}/{c['miss']})")
        self._lib_acc[h] = float(diag["acc"])
        ep = downsample(obs, act, rew)
        self._lib_mem[h] = ep
        return ep

    def library(self):
        """Your replays always count; the bot's own plays only once they're decent (--min-bot-acc),
        so it never studies its own bad habits."""
        files = self.library_files()
        eps = []
        for p in files:
            e = self.library_episode(p)
            if e is None:
                continue
            if p.parent == BOT_LIB and self._lib_acc.get(file_md5(p), 0.0) < self.a.min_bot_acc:
                continue
            eps.append(e)
        sig = sorted(file_md5(p) for p in files)
        keep = set(file_md5(p) for p in files)
        for h in [h for h in self._lib_mem if h not in keep]:     # replays that fell out of the library
            del self._lib_mem[h]
        return eps, sig, files

    def add_bot_play(self, replay_path: str, map_name: str):
        BOT_LIB.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in " -_[]()" else "_" for ch in map_name)[:70]
        try:
            shutil.copy(replay_path, BOT_LIB / f"{time.strftime('%Y%m%d_%H%M%S')} {safe}.osr")
        except OSError:
            pass

    def study(self, steps: int, reason: str) -> bool:
        """Offline RL over the whole replay library, then save the policy."""
        a = self.a
        eps, sig, files = self.library()
        if not eps:
            self.log(f"library is empty - put your own good .osr files in {LIB_DIR.resolve()}")
            return False
        n = sum(len(e[0]) for e in eps) * LIB_STRIDE
        mine = sum(1 for p in files if p.parent == LIB_DIR)
        self.log(f"{reason}: learning from {len(eps)} replays ({mine} yours, {len(files) - mine} the bot's) "
                 f"= {n / self.cfg.policy.hz / 60:.1f} min of play, {steps} steps")
        model = load_policy(a.policy, self.cfg.policy, self.dev)
        st0 = self._load_state()
        t0 = time.time()
        crit = steps // 4 if not st0.get("studied") else 0      # very first study: fit the critic first
        st = awr_train(model, eps, self.cfg, self.dev, steps, a.awr_lr, a.awr_beta, critic_steps=crit,
                       log=self.log if steps >= 1000 else None)
        if st is None or not all(torch.isfinite(p).all() for p in model.parameters()):
            self.log("study produced NaNs - discarded")
            return False
        self._save_policy(model, {"stage": "library_study", "library": len(eps)})
        st0.update(studied=True, lib_sig=sig)
        self._save_state(st0)
        self.log(f"{reason} done in {time.time() - t0:.0f}s (imitation-weighted loss {st['actor']:.2f}, "
                 f"value err {st['vf']:.3f}) -> saved {a.policy}")
        return True

    def _load_state(self) -> dict:
        if STATE_PATH.exists():
            try:
                return torch.load(STATE_PATH, map_location="cpu", weights_only=False)
            except Exception:  # noqa
                pass
        return {}

    def _save_state(self, st: dict):
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(st, STATE_PATH)

    def _save_policy(self, model, extra: dict):
        a = self.a
        HIST_DIR.mkdir(parents=True, exist_ok=True)
        orig = HIST_DIR / f"{Path(a.policy).stem}_ORIGINAL.pt"
        if not orig.exists():
            shutil.copy(a.policy, orig)                    # the policy as it was before any learning, kept forever
        shutil.copy(a.policy, HIST_DIR / f"{Path(a.policy).stem}_{time.strftime('%Y%m%d_%H%M%S')}.pt")
        olds = sorted(p for p in HIST_DIR.glob(f"{Path(a.policy).stem}_2*.pt"))
        for p in olds[:-a.keep_history]:
            p.unlink()
        tmp = a.policy + ".tmp"
        save_policy(tmp, model, extra)
        os.replace(tmp, a.policy)                          # atomic: the running bot never reads half a file

    def fresh_start(self):
        """New, untrained policy -> study ONLY the replay library (forgets everything learned before)."""
        from policy import ActorCritic
        a = self.a
        HIST_DIR.mkdir(parents=True, exist_ok=True)
        if os.path.exists(a.policy):
            bak = HIST_DIR / f"{Path(a.policy).stem}_before_fresh_{time.strftime('%Y%m%d_%H%M%S')}.pt"
            shutil.copy(a.policy, bak)
            self.log(f"old policy backed up to {bak}")
        Path(a.policy).parent.mkdir(parents=True, exist_ok=True)
        save_policy(a.policy, ActorCritic(self.cfg.policy), {"stage": "fresh"})
        if STATE_PATH.exists():
            STATE_PATH.unlink()                            # new critic + optimizer, warm-up runs again
        self.study(max(a.study_steps, 6000), "FRESH START (new policy, learning only from the replay library)")

    def startup_study(self):
        st = self._load_state()
        files = self.library_files()
        sig = sorted(file_md5(p) for p in files)
        if not files:
            self.log(f"replay library is empty - put your own good plays (.osr) in {LIB_DIR.resolve()}")
            return
        if st.get("lib_sig") == sig:
            self.log(f"replay library unchanged ({len(files)} replays) - no study needed")
            return
        self.study(self.a.study_steps, "STUDY SESSION (new replays in the library)")

    def make_learner(self, model):
        from ppo import PPOLearner
        a = self.a
        lr = PPOLearner(model, self.cfg, self.dev, lr=a.lr, bc_data=self.human, bc_coef=a.bc_coef,
                        epochs=a.epochs, minibatch=a.minibatch, ent_coef=a.ent, critic_warmup=a.warmup,
                        target_kl=a.target_kl)
        st = self._load_state()
        if "opt" in st:
            try:
                lr.load_state_dict(st)
            except Exception as e:  # noqa
                self.log(f"could not load {STATE_PATH} ({e}); starting a fresh optimizer")
        for g in lr.opt.param_groups:
            g["lr"] = a.lr
        return lr

    def update(self, ep, rewards):
        a, pc = self.a, self.cfg.policy
        if not os.path.exists(a.policy):
            raise SystemExit(f"{a.policy} not found - train a policy first (train_bc.py)")
        model = load_policy(a.policy, pc, self.dev)
        learner = self.make_learner(model)
        obs = torch.from_numpy(ep["obs"]).to(self.dev)
        with torch.no_grad():
            val = model(obs)[3].cpu().numpy().astype(np.float32)       # current critic, not the one used live
        T = len(rewards)
        done = np.zeros(T, np.float32); done[-1] = 1.0
        st = learner.update(ep["obs"][:, None], ep["act"][:, None], ep["logp"][:, None], val[:, None],
                            rewards[:, None].astype(np.float32), done[:, None], np.zeros(1, np.float32))
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            self.log("update produced NaNs - discarded, policy unchanged")
            return None
        lib = None
        if self.a.awr_steps > 0:                           # refresher over the whole replay library
            eps, sig, files = self.library()
            if eps:
                lib = awr_train(model, eps, self.cfg, self.dev, self.a.awr_steps, self.a.awr_lr, self.a.awr_beta,
                                log=None)
                lib = (lib, len(eps)) if lib else None
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    self.log("library refresher produced NaNs - discarded, policy unchanged")
                    return None
        self._save_policy(model, {"stage": "rl_after_map", "updates": learner.iters})
        state = self._load_state()
        state.update(learner.state_dict())
        if lib:
            state["lib_sig"] = sorted(file_md5(p) for p in self.library_files())
            state["studied"] = True
        self._save_state(state)
        st["library"] = lib
        return st, learner.iters

    def process(self, ep_path: Path, replay_path=None, allow_fallback=True) -> bool:
        """Returns True when the episode is finished with (learned from or discarded)."""
        ep = dict(np.load(ep_path, allow_pickle=False))
        if int(ep["n_objs"]) != self.cfg.policy.n_objs or int(ep["hz"]) != self.cfg.policy.hz:
            self.log(f"{ep_path.name}: recorded with a different n_objs/hz - discarded")
            return True
        res = self.rewards_for(ep, replay_path)
        if res is None or isinstance(res, str):
            if not allow_fallback:
                return False
            if not self.a.live_fallback:
                why = {None: "osu! saved no replay (failed / quit - use NoFail)", "bad_mods": "replay has speed mods",
                       "no_map": "beatmap not in Songs"}.get(res, "no replay")
                self.log(f"{ep_path.name}: {why} - SKIPPED (only osu!'s own score is used for learning)")
                return True
            c = ep["live_counts"]
            r, source = ep["rew_live"], "live"
            diag = {"counts": dict(zip(("300", "100", "50", "miss"), map(int, c[:4]))), "acc": float(ep["live_acc"]),
                    "offset_mean_ms": 0.0, "offset_std_ms": 0.0, "too_early": 0, "aim_miss": 0,
                    "no_note": int(c[4]), "w300_ms": hit_windows(float(ep["od"]))[0] * 1000}
            info = {"replay": "", "map": str(ep["map_name"]), "ar": float(ep["ar"]), "od": float(ep["od"]), "cs": ""}
            why = {None: "no replay found (failed / quit map, or osu!lazer)", "bad_mods": "replay has speed mods",
                   "no_map": "beatmap not in Songs"}[res]
            self.log(f"{why} -> learning from the bot's OWN judgement of this play (noisier)")
        else:
            r, source, diag, info = res
        t0 = time.time()
        out = self.update(ep, r)
        print("\n" + "=" * 78, flush=True)
        self.log(f"MAP: {info['map']}  (AR {info['ar']:.1f} OD {info['od']:.1f})  scored from: "
                 f"{'osu! replay ' + info['replay'] if source == 'replay' else 'live self-judgement'}")
        if source == "replay":
            self.log(f"replay matched: {info['matched']}")
        for line in describe(diag):
            self.log("  " + line)
        if info.get("warning"):
            self.log("  WARNING: " + info["warning"])
        self.log(f"  reward this map: {float(r.sum()):.1f} over {len(r)} steps")
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "map": info["map"], "source": source,
               "ar": round(float(info["ar"]), 2), "od": round(float(info["od"]), 2),
               "acc": round(diag["acc"], 4),
               **{f"n{k}": (diag.get("official") or diag["counts"])[k] for k in ("300", "100", "50", "miss")},
               "aim_miss": diag["aim_miss"], "too_early": diag["too_early"], "no_note": diag["no_note"],
               "offset_ms": round(diag["offset_mean_ms"], 1), "spread_ms": round(diag["offset_std_ms"], 1),
               "slider_follow": "" if diag.get("slider_follow") is None else round(diag["slider_follow"], 3),
               "reward": round(float(r.sum()), 1), "update": "", "pg": "", "vf": "", "kl": ""}
        if out is not None:
            st, n_up = out
            what = ("PPO update (policy improved)" if st["actor_on"] else
                    f"critic warm-up {n_up}/{self.a.warmup} (learning to predict reward before changing the policy)")
            self.log(f"  {what} in {time.time() - t0:.1f}s | pg {st['pg']:+.4f} vf {st['vf']:.3f} "
                     f"kl {st['kl']:+.4f}{' (early stop)' if st['early_stop'] else ''} -> saved {self.a.policy}")
            if st.get("library"):
                lst, n_lib = st["library"]
                self.log(f"  + studied the replay library ({n_lib} replays, {self.a.awr_steps} steps, "
                         f"value err {lst['vf']:.3f})")
            row.update(update=n_up, pg=round(st["pg"], 4), vf=round(st["vf"], 4), kl=round(st["kl"], 5))
        print("=" * 78, flush=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        new = not LOG_PATH.exists()
        with open(LOG_PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        USED_PATH.parent.mkdir(parents=True, exist_ok=True)
        USED_PATH.write_text(json.dumps(sorted(self.used)[-2000:]))
        return True

    def finish(self, ep_path: Path):
        DONE_DIR.mkdir(parents=True, exist_ok=True)
        os.replace(ep_path, DONE_DIR / ep_path.name)
        for p in sorted(DONE_DIR.glob("*.npz"))[:-self.a.keep_episodes]:
            p.unlink()

    def pending(self):
        return sorted(p for p in EP_DIR.glob("*.npz") if not p.name.endswith(".tmp.npz"))

    def run_once(self, wait_for_replay: float):
        for ep_path in self.pending():
            try:
                age = time.time() - ep_path.stat().st_mtime
                done = self.process(ep_path, allow_fallback=age > wait_for_replay)
                if done:
                    self.finish(ep_path)
            except Exception as e:  # noqa
                import traceback
                self.log(f"error on {ep_path.name}: {type(e).__name__}: {e}")
                traceback.print_exc()
                self.finish(ep_path)


# ====================================================================== misc commands
def rollback(policy):
    files = sorted(HIST_DIR.glob(f"{Path(policy).stem}_2*.pt"))
    if not files:
        raise SystemExit("No backups in checkpoints/history.")
    shutil.copy(files[-1], policy)
    files[-1].unlink()
    print(f"restored {policy} from {files[-1].name}")


def stats():
    if not LOG_PATH.exists():
        raise SystemExit("No maps learned yet.")
    rows = list(csv.DictReader(open(LOG_PATH)))
    print(f"{'#':>4} {'time':19} {'src':6} {'AR':>4} {'OD':>4} {'acc':>6} {'miss':>4} {'aim':>4} {'early':>5} "
          f"{'offset':>7} {'reward':>7}  map")
    for i, r in enumerate(rows, 1):
        print(f"{i:4d} {r['time']:19} {r['source']:6} {r['ar']:>4} {r['od']:>4} {float(r['acc']) * 100:5.1f}% "
              f"{r['nmiss']:>4} {r['aim_miss']:>4} {r['too_early']:>5} {r['offset_ms']:>7} {r['reward']:>7}  {r['map'][:30]}")
    accs = [float(r["acc"]) for r in rows if r["source"] == "replay"]
    if len(accs) >= 10:
        k = len(accs) // 3
        print(f"\nfirst {k} maps avg acc {np.mean(accs[:k]) * 100:.1f}%  ->  last {k} maps {np.mean(accs[-k:]) * 100:.1f}%"
              "  (different maps differ in difficulty, so compare over many plays)")


PARENT_GONE = threading.Event()


def _watch_parent():
    """run_bot.py holds our stdin open; when the bot closes (even via the kill switch) the pipe closes.
    We then finish learning from any play still waiting, and exit."""
    try:
        while sys.stdin.buffer.read(1024):
            pass
    except Exception:  # noqa
        pass
    PARENT_GONE.set()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--watch", action="store_true", help="keep running; learn after every finished map")
    g.add_argument("--once", action="store_true", help="process waiting episodes and exit")
    g.add_argument("--episode", help="one episode file (use with --replay)")
    g.add_argument("--rollback", action="store_true", help="restore the policy from before the last update")
    g.add_argument("--stats", action="store_true", help="print the per-map learning log")
    g.add_argument("--study", action="store_true", help="study the replay library now, then exit (no osu! needed)")
    g.add_argument("--fresh", action="store_true", help="replace the policy with a NEW one learned only from the library")
    ap.add_argument("--replay", default=None, help="replay for --episode (otherwise it is searched for)")
    ap.add_argument("--policy", default="checkpoints/policy.pt", help="the policy the bot plays with (updated in place)")
    ap.add_argument("--songs", default=DEFAULT_SONGS)
    ap.add_argument("--replay-dirs", nargs="+", default=DEFAULT_REPLAY_DIRS)
    ap.add_argument("--extra-songs", nargs="*", default=["songs"], help="more folders with .osu files (project songs/)")
    ap.add_argument("--max-replays", type=int, default=300, help="max replays in the library (yours first, then the bot's)")
    ap.add_argument("--study-steps", type=int, default=10000, help="study session length when the library changes")
    ap.add_argument("--awr-steps", type=int, default=1500, help="library study after every map (0 = off)")
    ap.add_argument("--awr-lr", type=float, default=3e-4)
    ap.add_argument("--min-bot-acc", type=float, default=0.9,
                    help="the bot's own plays are only studied once they reach this accuracy")
    ap.add_argument("--awr-beta", type=float, default=1.0, help="lower = copy only the very best moments")
    ap.add_argument("--human-data", default="", help="(old) npz dataset mixed into PPO; the replay library replaces it")
    ap.add_argument("--bc-coef", type=float, default=0.5, help="weight of --human-data in PPO (if given)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=6, help="PPO passes over each map's data")
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--target-kl", type=float, default=0.02, help="stop an update once the policy moved this much")
    ap.add_argument("--ent", type=float, default=0.0, help="entropy bonus (exploration)")
    ap.add_argument("--warmup", type=int, default=3, help="first N maps only train the critic")
    ap.add_argument("--wait-replay", type=float, default=45.0,
                    help="seconds to wait for osu! to save the replay before falling back to the live judgement")
    ap.add_argument("--live-fallback", dest="live_fallback", action="store_true",
                    help="learn from the bot's OWN judgement when osu! saved no replay (off: such maps are skipped)")
    ap.add_argument("--no-live-fallback", dest="live_fallback", action="store_false", help=argparse.SUPPRESS)
    ap.add_argument("--keep-history", type=int, default=30)
    ap.add_argument("--keep-episodes", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--exit-on-stdin-eof", action="store_true", help="used by run_bot.py")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.exit_on_stdin_eof:
        threading.Thread(target=_watch_parent, daemon=True).start()
    if a.rollback:
        return rollback(a.policy)
    if a.stats:
        return stats()

    w = Worker(a)
    if a.study:
        w.study(a.study_steps, "STUDY SESSION")
        return
    if a.fresh:
        w.fresh_start()
        return
    if a.episode:
        w.process(Path(a.episode), a.replay)
        return
    if a.once:
        for p in w.pending():
            if w.process(p, allow_fallback=True):
                w.finish(p)
        return
    w.log(f"watching {EP_DIR} for finished maps; replays from {a.replay_dirs}")
    w.log(f"policy: {a.policy} (updated after every map, backups in {HIST_DIR})")
    w.log(f"replay library: {LIB_DIR.resolve()} (max {a.max_replays} replays)")
    try:
        w.startup_study()
    except Exception as e:  # noqa
        import traceback
        w.log(f"study failed: {type(e).__name__}: {e}"); traceback.print_exc()
    gone_at = None
    try:
        while True:
            w.run_once(a.wait_replay)
            if PARENT_GONE.is_set():
                if gone_at is None:
                    gone_at = time.time()
                    if w.pending():
                        w.log("bot closed - finishing the learning from the last play before exiting...")
                if not w.pending() or time.time() - gone_at > a.wait_replay + 90:
                    w.log("learning finished - worker exiting (the next run_bot loads the improved policy)")
                    return
            time.sleep(2.0)
    except KeyboardInterrupt:
        w.log("stopped")


if __name__ == "__main__":
    main()
