"""Turn human replays (.osr) + their beatmaps (.osu) into (observation, action) training pairs.

The beatmap is used HERE ONLY, to reconstruct what the player could see on screen at each moment
(notes that had appeared and not yet been hit/missed). The policy that gets trained never reads a map.

    python replay_data.py --replays "C:/Users/you/AppData/Local/osu!/Data/r" \
                          --songs   "C:/Users/you/AppData/Local/osu!/Songs"  --out data/replays.npz

Skips replays with DT/HT/Relax/Auto and (optionally) low accuracy. HR/EZ are handled.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from config import BotConfig, ar_to_preempt
from judge import Judge, hit_windows
from obs import VisObj, VisBall, CIRCLE, SLIDER_HEAD, SPINNER as SPIN_KIND, encode, obs_dim, sort_visible
from osu_formats import (Beatmap, Replay, parse_osr, parse_osu, BAD_MODS, SLIDER, SPINNER)


# ====================================================================== beatmap index
def index_songs(songs_dir: str, cache: str = "data/md5_cache.json") -> dict:
    """md5 of each .osu file -> path (cached; first run over a big Songs folder takes a while)."""
    cache_p = Path(cache)
    known = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    by_path = {v: k for k, v in known.items()}
    files = list(Path(songs_dir).rglob("*.osu"))
    new = 0
    for p in files:
        sp = str(p)
        if sp in by_path:
            continue
        known[hashlib.md5(p.read_bytes()).hexdigest()] = sp
        new += 1
    if new:
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        cache_p.write_text(json.dumps(known))
    print(f"[index] {len(files)} beatmap files ({new} newly hashed)")
    return known


# ====================================================================== judge a human replay
def judge_replay(bm: Beatmap, rep: Replay) -> Judge:
    j = Judge(bm.objects, bm.od, bm.radius)
    presses = []
    for ki in (0, 1):
        s = rep.key_state(ki)
        prev = np.concatenate([[False], s[:-1]])
        for k in np.nonzero(s & ~prev)[0]:
            presses.append((rep.t[k], rep.x[k], rep.y[k]))
    presses.sort()
    last_t = -1e9
    for t, x, y in presses:
        if t - last_t < 1e-4:       # both keys in the same frame = one click
            continue
        j.press(t, (x, y))
        last_t = t
    j.advance(1e9)
    # slider-follow / spinner stats on a 5 ms grid
    g = np.arange(rep.t[0], rep.t[-1], 0.005)
    gx, gy = np.interp(g, rep.t, rep.x), np.interp(g, rep.t, rep.y)
    held = (rep.key_state(0) | rep.key_state(1))
    gk = held[np.clip(np.searchsorted(rep.t, g, side="right") - 1, 0, len(held) - 1)]
    for t, x, y, k in zip(g, gx, gy, gk):
        j.track(t, 0.005, (x, y), bool(k))
    j.finalize()
    return j


# ====================================================================== build samples
def build_samples(bm: Beatmap, rep: Replay, judge: Judge, cfg: BotConfig):
    pc = cfg.policy
    hz, dt = pc.hz, 1.0 / pc.hz
    preempt = ar_to_preempt(bm.ar)
    objs = bm.objects
    t0 = objs[0].time - preempt - 0.3
    t1 = max(o.end_time for o in objs) + 0.5
    grid = np.arange(t0, t1, dt)
    n = len(grid)
    px = np.interp(grid, rep.t, rep.x)
    py = np.interp(grid, rep.t, rep.y)
    idx = np.clip(np.searchsorted(rep.t, grid, side="right") - 1, 0, len(rep.t) - 1)
    k0 = rep.key_state(0)[idx]
    k1 = rep.key_state(1)[idx]
    outside = (grid < rep.t[0]) | (grid > rep.t[-1])
    k0 = k0 & ~outside
    k1 = k1 & ~outside

    obs = np.zeros((n - 1, obs_dim(pc.n_objs)), np.float32)
    act = np.zeros((n - 1, 4), np.float32)
    ptr, live = 0, []
    age = [1.0, 1.0]
    sliders = [o for o in objs if o.kind == SLIDER]
    r = bm.radius
    for k in range(n - 1):
        t = grid[k]
        while ptr < len(objs) and objs[ptr].time - preempt <= t:
            live.append(objs[ptr]); ptr += 1
        keep, vis = [], []
        for o in live:
            if o.kind == SPINNER:
                if t > o.end_time:
                    continue
                kind = SPIN_KIND
            else:
                if judge.resolved.get(o.index, 1e18) <= t:
                    continue
                kind = SLIDER_HEAD if o.kind == SLIDER else CIRCLE
            keep.append(o)
            vis.append(VisObj(o.index, kind, o.x, o.y, r, o.time - t))
        live = keep
        ball = None
        for s in sliders:
            if s.time <= t <= s.end_time:
                b, b2 = s.ball_pos(t), s.ball_pos(t + 0.01)
                v = (b2 - b) / 0.01
                ball = VisBall(float(b[0]), float(b[1]), float(v[0]), float(v[1]))
                break
        if k > 0:
            vel = ((px[k] - px[k - 1]) / dt, (py[k] - py[k - 1]) / dt)
            for i, (a_, b_) in enumerate(((k0[k - 1], k0[k]), (k1[k - 1], k1[k]))):
                age[i] = 0.0 if a_ != b_ else age[i] + dt
        else:
            vel = (0.0, 0.0)
        encode(pc.n_objs, (px[k], py[k]), vel, (bool(k0[k]), bool(k1[k])), age,
               sort_visible(vis), ball, preempt, out=obs[k])
        act[k, 0] = (px[k + 1] - px[k]) / dt / pc.max_speed
        act[k, 1] = (py[k + 1] - py[k]) / dt / pc.max_speed
        act[k, 2], act[k, 3] = float(k0[k + 1]), float(k1[k + 1])
    np.clip(act[:, :2], -4.0, 4.0, out=act[:, :2])
    return obs, act


def process_pair(osr_path: str, osu_path: str, cfg: BotConfig, min_acc: float):
    rep = parse_osr(osr_path)
    if rep.mode != 0:
        return None, "not osu!standard"
    if rep.mods & BAD_MODS:
        return None, f"unsupported mods ({rep.mods})"
    bm = parse_osu(osu_path).with_mods(rep.mods)
    if len(bm.objects) < 10:
        return None, "map too small"
    judge = judge_replay(bm, rep)
    acc = judge.accuracy()
    if acc < min_acc:
        return None, f"accuracy {acc:.3f} < {min_acc}"
    obs, act = build_samples(bm, rep, judge, cfg)
    return (obs, act, judge.counts, acc), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", required=True, help="folder with .osr files (osu!/Data/r)")
    ap.add_argument("--songs", required=True, help="folder with beatmaps (osu!/Songs)")
    ap.add_argument("--out", default="data/replays.npz")
    ap.add_argument("--min-acc", type=float, default=0.90, help="skip replays that judge below this accuracy")
    ap.add_argument("--max-replays", type=int, default=500)
    a = ap.parse_args()

    cfg = BotConfig()
    idx = index_songs(a.songs)
    all_obs, all_act, rows = [], [], []
    files = sorted(Path(a.replays).rglob("*.osr"))[:a.max_replays]
    for f in files:
        try:
            rep = parse_osr(f)
            osu = idx.get(rep.beatmap_md5)
            if osu is None:
                print(f"  skip {f.name}: beatmap not found in Songs"); continue
            res, why = process_pair(str(f), osu, cfg, a.min_acc)
            if res is None:
                print(f"  skip {f.name}: {why}"); continue
            obs, act, counts, acc = res
            all_obs.append(obs); all_act.append(act)
            print(f"  ok   {f.name}: {len(obs)} steps, judged acc {acc * 100:.1f}% {counts}")
        except Exception as e:  # noqa
            print(f"  skip {f.name}: {type(e).__name__}: {e}")
    if not all_obs:
        raise SystemExit("No usable replays. Check paths / mods / --min-acc.")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, obs=np.concatenate(all_obs), act=np.concatenate(all_act),
                        n_objs=cfg.policy.n_objs, hz=cfg.policy.hz)
    print(f"[data] wrote {a.out}: {sum(len(o) for o in all_obs)} samples from {len(all_obs)} replays")


if __name__ == "__main__":
    main()
