"""Smoke test of the whole learning pipeline with NO osu! install and no real replays:

  scripted "expert" plays simulator maps -> written out as real .osu / .osr files ->
  parsed back -> judged -> turned into a dataset -> behaviour-cloned -> evaluated in the simulator.

    python selftest.py            (about a minute on CPU)
"""
from __future__ import annotations
import argparse
import lzma
import math
import struct
import tempfile
from pathlib import Path

import numpy as np
import torch

from config import BotConfig
from map_gen import random_beatmap
from osu_formats import parse_osu, parse_osr, CIRCLE, SLIDER, SPINNER
from replay_data import process_pair
from sim_env import OsuSimEnv


# ---------------------------------------------------------------- writers (test only)
def write_osu(bm, path: Path):
    lines = ["osu file format v14", "", "[Difficulty]", f"HPDrainRate:5", f"CircleSize:{bm.cs:.2f}",
             f"OverallDifficulty:{bm.od:.2f}", f"ApproachRate:{bm.ar:.2f}", f"SliderMultiplier:{bm.slider_multiplier}",
             "SliderTickRate:1", "", "[TimingPoints]", "0,500,4,1,0,100,1,0", "", "[HitObjects]"]
    for o in bm.objects:
        t = int(round(o.time * 1000))
        if o.kind == CIRCLE:
            lines.append(f"{int(o.x)},{int(o.y)},{t},1,0")
        elif o.kind == SPINNER:
            lines.append(f"256,192,{t},8,0,{int(round(o.end_time * 1000))}")
        else:
            # linear slider with the generated length (beat length fixed to 500ms in this test file)
            length = bm.slider_multiplier * 100.0 * (o.end_time - o.time) / 0.5 / o.slides
            p0, p1 = o.path[0], o.path[-1]
            d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-6)
            end = p0 + d * length
            lines.append(f"{int(p0[0])},{int(p0[1])},{t},2,0,L|{int(end[0])}:{int(end[1])},{o.slides},{length:.2f}")
    path.write_text("\n".join(lines))


def _s(txt: str) -> bytes:
    b = txt.encode()
    out = bytearray([0x0B]); n = len(b)
    while True:
        c = n & 0x7F; n >>= 7
        out.append(c | (0x80 if n else 0))
        if not n:
            break
    return bytes(out) + b


def write_osr(path: Path, frames, md5="x" * 32, mods=0):
    txt = "0|256|-500|0,-1|256|-500|0," + ",".join(f"{w}|{x:.2f}|{y:.2f}|{z}" for w, x, y, z in frames) + ",-12345|0|0|1234"
    comp = lzma.compress(txt.encode(), format=lzma.FORMAT_ALONE)
    b = bytearray([0]) + struct.pack("<i", 20200101) + _s(md5) + _s("tester") + _s("r" * 32)
    b += struct.pack("<6h", 0, 0, 0, 0, 0, 0) + struct.pack("<ihb", 0, 0, 0) + struct.pack("<i", mods)
    b += _s("") + struct.pack("<q", 0) + struct.pack("<i", len(comp)) + comp + struct.pack("<q", 0)
    path.write_bytes(bytes(b))


# ---------------------------------------------------------------- scripted expert (stands in for a human)
def expert_play(env: OsuSimEnv, bm, rng, dt_ms=8):
    env.noise = False
    env.reset(bm)
    frames, last_t = [], -0.001   # replay clock: previous dummy frame sits at -1 ms
    hold_until, hold_key, tap_off = -1.0, 0, 0.0
    prev_target = None
    while True:
        vis = env._visible_true()
        taps = [v for v in vis if v.kind != 2]
        ball = env._ball()
        a = np.zeros(4)
        if ball is not None:
            tgt = np.array([ball.x + ball.vx * 0.02, ball.y + ball.vy * 0.02]); a[2 + hold_key] = 1
            T = 0.06
        elif taps:
            o = taps[0]
            tgt = np.array([o.x, o.y]) + rng.normal(0, 2.0, 2) * 0
            if prev_target != o.id:
                prev_target, tap_off = o.id, float(rng.normal(0.004, 0.012))
                hold_key = 1 - hold_key if (taps[1].ttl - o.ttl < 0.25 if len(taps) > 1 else False) else 0
            T = max(0.03, o.ttl - 0.02)
            if o.ttl <= -tap_off and o.ttl > -0.05:
                a[2 + hold_key] = 1
                hold_until = env.t + 0.04
        elif vis and vis[0].kind == 2:
            ang = env.t * 30
            tgt = np.array([256 + 90 * math.cos(ang), 192 + 90 * math.sin(ang)]); T = 0.05
        else:
            tgt = env.pos; T = 0.1
        if env.t < hold_until:
            a[2 + hold_key] = 1
        a[:2] = np.clip((tgt - env.pos) / T / env.pc.max_speed, -3, 3)
        _, _, done, info = env.step(a)
        w = int(round((env.t - last_t) * 1000)); last_t = last_t + w / 1000.0
        z = (1 if env.keys[0] else 0) | (2 if env.keys[1] else 0)
        frames.append((max(w, 1), env.pos[0], env.pos[1], z))
        if done:
            return frames, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--save-data", default=None, help="also write the dataset (npz) so you can try train_bc.py / train_rl.py")
    a = ap.parse_args()
    from policy import ActorCritic, bc_loss, augment
    from sim_eval import eval_policy

    cfg = BotConfig()
    rng = np.random.default_rng(0)
    env = OsuSimEnv(cfg, seed=0, perception_noise=False)
    tmp = Path(tempfile.mkdtemp())
    all_o, all_a = [], []
    for i in range(a.maps):
        bm0 = random_beatmap(rng, 40.0, difficulty=0.3 + 0.1 * i)
        osu = tmp / f"m{i}.osu"
        write_osu(bm0, osu)
        bm = parse_osu(osu)                                       # also tests the .osu parser
        assert len(bm.objects) == len(bm0.objects)
        frames, info = expert_play(env, bm, rng)
        osr = tmp / f"r{i}.osr"
        write_osr(osr, frames)
        rep = parse_osr(osr)                                      # and the .osr parser (LZMA etc.)
        res, why = process_pair(str(osr), str(osu), cfg, min_acc=0.0)
        obs, act, counts, acc = res
        print(f"map {i}: {len(bm.objects)} objects | sim says {info['counts']} | replay judge says {counts} | "
              f"{len(obs)} samples")
        all_o.append(obs); all_a.append(act)
    if a.save_data:
        import os
        os.makedirs(os.path.dirname(a.save_data) or ".", exist_ok=True)
        np.savez_compressed(a.save_data, obs=np.concatenate(all_o), act=np.concatenate(all_a),
                            n_objs=cfg.policy.n_objs, hz=cfg.policy.hz)
    obs = torch.from_numpy(np.concatenate(all_o)); act = torch.from_numpy(np.concatenate(all_a))
    print(f"dataset: {obs.shape}, key-down fraction {act[:, 2:].mean():.3f}")

    model = ActorCritic(cfg.policy)
    opt = torch.optim.AdamW(model.parameters(), 2e-3)
    perm_n = len(obs)
    for ep in range(a.epochs):
        p = torch.randperm(perm_n)
        tot = 0; nb = 0
        for s in range(0, perm_n - 1024, 1024):
            i = p[s:s + 1024]
            o, y = augment(obs[i], act[i], cfg.policy)
            loss, parts = bc_loss(model, o, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += parts["nll"]; nb += 1
        print(f"  epoch {ep + 1}: nll {tot / nb:.3f} key-acc {parts['key_acc']:.3f}")
    r = eval_policy(model, cfg, None, 2, max_seconds=20)
    print("sim play after BC:", r)
    print("SELFTEST DONE")


if __name__ == "__main__":
    main()
