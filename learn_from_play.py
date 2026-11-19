"""(SUPERSEDED by rl_worker.py, which run_bot.py --policy now uses: true PPO reinforcement learning after every map.
 Don't run this at the same time as the bot - both would edit the policy.)

OLD: reward-weighted imitation of the bot's OWN finished plays.

    python learn_from_play.py --watch        # leave running next to the bot; learns after every finished map
    python learn_from_play.py --latest       # learn from the newest replay once
    python learn_from_play.py --replay some.osr
    python learn_from_play.py --latest --score-only    # just analyse + print the reward, no learning
    python learn_from_play.py --rollback     # undo the last policy update

What happens after each map
  1. osu! saves the replay (.osr). We wait for the file to finish writing.
  2. The replay is matched to its beatmap and judged with YOUR judge.py (same rules as RL), giving every
     note a 300 / 100 / 50 / miss plus a reward:  r300=1.0, r100=0.5, r50=0.2, miss=0 (config.RewardConfig),
     plus the timing bonus for a tap close to 0 ms.
  3. Every 1/120 s step of the play is tagged with the reward of the note it was aiming at. Steps aimed at a
     missed note get weight 0, a 50 gets little, a 100 some, a clean 300 the most.
  4. The policy takes a SMALL reward-weighted imitation step on those steps (mixed with your human replays so it
     does not drift), i.e. "do more of what scored well".
  5. Safety: the old policy is backed up first, and the update is kept only if the simulator score does not get
     worse. Everything is logged to data/self_play/log.csv.

Note: this reinforces behaviour that already worked. It cannot invent new behaviour (that is what train_rl.py
is for), so expect small, steady gains per play, not jumps.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
from osu_formats import parse_osr, parse_osu, BAD_MODS, SPINNER, SLIDER
from policy import load_policy, save_policy, bc_loss, augment
from replay_data import index_songs, judge_replay, build_samples
from sim_eval import eval_policy

LOCAL = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "osu!"
DEFAULT_SONGS = str(LOCAL / "Songs")
# osu!stable auto-saves finished plays in Data\r ; the Replays folder holds replays you export.
DEFAULT_REPLAY_DIRS = [str(LOCAL / "Replays"), str(LOCAL / "Data" / "r")]
SP_DIR = Path("data/self_play")
BUF_DIR = SP_DIR / "buffer"
HIST_DIR = Path("checkpoints/history")


# ====================================================================== scoring one play
def hit_windows(od: float):
    return (80 - 6 * od) / 1000.0, (140 - 8 * od) / 1000.0, (200 - 10 * od) / 1000.0


def note_outcomes(bm, judge, rc):
    """-> {object index: (label, timing_error_s or None, reward)} using osu!'s windows + RewardConfig."""
    w300, w100, w50 = hit_windows(bm.od)
    dts = judge.dts
    if len(dts) and not any(o.index in dts for o in bm.objects):
        raise RuntimeError("judge.dts is not keyed by object index - send me judge.py so I can match it.")
    out = {}
    for o in bm.objects:
        if o.kind == SPINNER:                       # spinners can't be graded by a tap: neutral weight
            out[o.index] = ("spin", None, 0.5 * rc.r300)
            continue
        dt = dts.get(o.index)
        if dt is None or abs(dt) > w50:
            out[o.index] = ("miss", dt, rc.miss)
            continue
        a = abs(dt)
        label, base = ("300", rc.r300) if a <= w300 else ("100", rc.r100) if a <= w100 else ("50", rc.r50)
        bonus = rc.timing_bonus * math.exp(-0.5 * (dt / rc.timing_sigma) ** 2)
        out[o.index] = (label, dt, base + bonus)
    return out


def step_weights(bm, judge, outcomes, t_steps, rc, power):
    """Weight in [0,1] for every training step = reward of the note that step was working toward."""
    pairs = sorted((judge.resolved[o.index], o.index) for o in bm.objects if o.index in judge.resolved)
    if not pairs:
        return np.zeros(len(t_steps), np.float32)
    res_t = np.array([p[0] for p in pairs])
    r_ref = rc.r300 + rc.timing_bonus
    w_obj = np.array([np.clip(outcomes[i][2] / r_ref, 0.0, 1.0) ** power for _, i in pairs])
    idx = np.searchsorted(res_t, t_steps, side="left")
    w = np.where(idx < len(res_t), w_obj[np.minimum(idx, len(res_t) - 1)], 0.0).astype(np.float32)
    for o in bm.objects:                              # slider-following / spinning belongs to its own note
        if o.kind in (SLIDER, SPINNER) and o.index in outcomes:
            m = (t_steps >= o.time) & (t_steps <= o.end_time)
            w[m] = np.clip(outcomes[o.index][2] / r_ref, 0.0, 1.0) ** power
    return w


def summarize(bm, outcomes, judge, rc):
    cnt = {"300": 0, "100": 0, "50": 0, "miss": 0, "spin": 0}
    total = 0.0
    for lab, _, r in outcomes.values():
        cnt[lab] += 1
        total += r
    n_tap = len(outcomes) - cnt["spin"]
    best = n_tap * (rc.r300 + rc.timing_bonus) + cnt["spin"] * 0.5 * rc.r300
    return cnt, total, best


# ====================================================================== data buffer
def save_to_buffer(name, obs, act, w):
    BUF_DIR.mkdir(parents=True, exist_ok=True)
    keep = w > 1e-3
    np.savez_compressed(BUF_DIR / f"{name}.npz", obs=obs[keep], act=act[keep], w=w[keep])
    return int(keep.sum())


def load_buffer(max_plays):
    files = sorted(BUF_DIR.glob("*.npz"), key=lambda p: p.stat().st_mtime)[-max_plays:]
    if not files:
        return None
    ds = [np.load(f) for f in files]
    return (np.concatenate([d["obs"] for d in ds]), np.concatenate([d["act"] for d in ds]),
            np.concatenate([d["w"] for d in ds]), len(files))


# ====================================================================== the (small) learning step
def finetune(model, cfg, s_obs, s_act, s_w, human, a, dev, rng):
    params = (list(model.actor.parameters()) + list(model.mu.parameters()) +
              list(model.logstd.parameters()) + list(model.keys.parameters()))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    p = (s_w / s_w.sum()).astype(np.float64)          # sample steps in proportion to their reward
    n_self = int(a.batch * a.mix) if human is not None else a.batch
    n_hum = a.batch - n_self if human is not None else 0
    model.train()
    last = {}
    for _ in range(a.steps):
        i = rng.choice(len(s_w), size=n_self, p=p)
        o, y = s_obs[i], s_act[i]
        if n_hum:
            j = rng.integers(0, len(human[0]), n_hum)
            o, y = np.concatenate([o, human[0][j]]), np.concatenate([y, human[1][j]])
        o, y = augment(torch.from_numpy(o).to(dev), torch.from_numpy(y).to(dev), cfg.policy)
        loss, last = bc_loss(model, o, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    model.eval()
    return last


def sim_score(model, cfg, bm, episodes):
    torch.manual_seed(1234)                           # same dice before and after -> fair comparison
    return eval_policy(model, cfg, [bm], episodes, seed=4321)


def update_policy(bm, a, cfg, dev, rng):
    if not os.path.exists(a.policy):
        print(f"[learn] {a.policy} not found - train a policy first (train_bc.py)")
        return None
    buf = load_buffer(a.buffer)
    if buf is None or buf[2].sum() <= 0:
        print("[learn] no rewarded steps yet - nothing to learn from")
        return None
    s_obs, s_act, s_w, n_plays = buf
    human = None
    if a.human_data and os.path.exists(a.human_data):
        d = np.load(a.human_data)
        if int(d["n_objs"]) == cfg.policy.n_objs:
            human = (d["obs"], d["act"])
    model = load_policy(a.policy, cfg.policy, dev)
    model.eval()
    before = sim_score(model, cfg, bm, a.eval_episodes)
    stats = finetune(model, cfg, s_obs, s_act, s_w, human, a, dev, rng)
    after = sim_score(model, cfg, bm, a.eval_episodes)
    ok = after["acc"] >= before["acc"] - a.max_drop
    if ok:
        HIST_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(a.policy, HIST_DIR / f"{Path(a.policy).stem}_{time.strftime('%Y%m%d_%H%M%S')}.pt")
        tmp = a.policy + ".tmp"                       # write then swap, so a running bot never reads half a file
        save_policy(tmp, model, {"stage": "self_play", "plays_in_buffer": n_plays})
        os.replace(tmp, a.policy)
    print(f"[learn] {len(s_w)} rewarded steps from the last {n_plays} play(s) | sim accuracy "
          f"{before['acc'] * 100:.1f}% -> {after['acc'] * 100:.1f}% | "
          f"{'SAVED' if ok else 'REJECTED (got worse) - policy unchanged'}")
    return before["acc"], after["acc"], ok


# ====================================================================== one replay end to end
def handle_replay(path, ctx):
    a, cfg = ctx["args"], ctx["cfg"]
    rep = parse_osr(path)
    if rep.mode != 0:
        print(f"[skip] {Path(path).name}: not osu!standard"); return
    if rep.mods & BAD_MODS:
        print(f"[skip] {Path(path).name}: unsupported mods ({rep.mods})"); return
    if a.player:
        who = getattr(rep, "player", None)
        if who is None:
            print("[warn] --player given but the replay parser exposes no player name; not filtering")
        elif who.lower() != a.player.lower():
            print(f"[skip] {Path(path).name}: played by {who}, not {a.player}"); return
    osu = ctx["idx"].get(rep.beatmap_md5)
    if osu is None:                                   # new map since last index?
        ctx["idx"] = index_songs(a.songs)
        osu = ctx["idx"].get(rep.beatmap_md5)
    if osu is None:
        print(f"[skip] {Path(path).name}: beatmap not found in {a.songs}"); return
    bm = parse_osu(osu).with_mods(rep.mods)
    if len(bm.objects) < 10:
        print(f"[skip] {Path(path).name}: map too small"); return

    judge = judge_replay(bm, rep)
    outcomes = note_outcomes(bm, judge, cfg.reward)
    cnt, total, best = summarize(bm, outcomes, judge, cfg.reward)
    print("\n" + "=" * 78)
    print(f"[play] {getattr(bm, 'title', Path(osu).stem)}")
    print(f"       judge.py: acc {judge.accuracy() * 100:.1f}%  {dict(judge.counts)}")
    print(f"       per-note: 300={cnt['300']} 100={cnt['100']} 50={cnt['50']} miss={cnt['miss']} (spinners {cnt['spin']})")
    print(f"       REWARD POINTS: {total:.1f} / {best:.1f}  ({100 * total / max(best, 1e-9):.1f}% of perfect)")
    for lab in ("300", "100", "50", "miss"):          # sanity check against judge.py's own totals
        want = judge.counts.get(lab, 0)
        if abs(cnt[lab] - want) > max(3, 0.05 * len(bm.objects)):
            print(f"       (note: my {lab} count {cnt[lab]} differs from judge.py's {want} - slider bodies are "
                  "scored by osu! differently from taps; weights use the tap timing)")
            break

    obs, act = build_samples(bm, rep, judge, cfg)
    dt = 1.0 / cfg.policy.hz
    t0 = bm.objects[0].time - ar_to_preempt(bm.ar) - 0.3
    t1 = max(o.end_time for o in bm.objects) + 0.5
    t_steps = np.arange(t0, t1, dt)[:len(obs)]
    w = step_weights(bm, judge, outcomes, t_steps, cfg.reward, a.power)
    w[t_steps > rep.t[-1]] = 0.0                      # play ended early (quit / fail): ignore the rest
    n_keep = save_to_buffer(Path(path).stem[:80], obs, act, w)
    print(f"       {n_keep} of {len(obs)} steps carry reward weight (misses / 50s are dropped)")

    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "replay": Path(path).name,
           "map": getattr(bm, "title", ""), "acc": round(judge.accuracy(), 4), "n300": cnt["300"],
           "n100": cnt["100"], "n50": cnt["50"], "miss": cnt["miss"], "reward": round(total, 1),
           "reward_pct": round(100 * total / max(best, 1e-9), 1), "sim_before": "", "sim_after": "", "kept": ""}
    if not a.score_only:
        res = update_policy(bm, a, cfg, ctx["dev"], ctx["rng"])
        if res:
            row.update(sim_before=round(res[0], 4), sim_after=round(res[1], 4), kept=res[2])
    SP_DIR.mkdir(parents=True, exist_ok=True)
    log = SP_DIR / "log.csv"
    new = not log.exists()
    with open(log, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(row))
        if new:
            wr.writeheader()
        wr.writerow(row)
    print("=" * 78)


# ====================================================================== finding replays
def list_replays(dirs):
    out = []
    for d in dirs:
        if os.path.isdir(d):
            out += [str(p) for p in Path(d).glob("*.osr")]
    return out


def file_hash(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def load_done():
    f = SP_DIR / "processed.json"
    return set(json.loads(f.read_text())) if f.exists() else set()


def save_done(done):
    SP_DIR.mkdir(parents=True, exist_ok=True)
    (SP_DIR / "processed.json").write_text(json.dumps(sorted(done)))


def process_once(path, ctx, done):
    try:
        h = file_hash(path)
        if h in done:
            return
        handle_replay(path, ctx)
        done.add(h)
        save_done(done)
    except Exception as e:  # noqa
        import traceback
        print(f"[error] {Path(path).name}: {type(e).__name__}: {e}")
        traceback.print_exc()


def rollback(a):
    files = sorted(HIST_DIR.glob(f"{Path(a.policy).stem}_*.pt"))
    if not files:
        raise SystemExit("No backups found in checkpoints/history.")
    shutil.copy(files[-1], a.policy)
    print(f"restored {a.policy} from {files[-1].name}")
    files[-1].unlink()


def _exit_when_parent_dies():
    """The bot holds our stdin pipe open; when the bot exits (even via the ESC kill switch) it closes."""
    try:
        while sys.stdin.buffer.read(1024):
            pass
    except Exception:  # noqa
        pass
    os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--watch", action="store_true", help="keep running; learn from every new finished play")
    g.add_argument("--latest", action="store_true", help="learn from the newest replay once")
    g.add_argument("--replay", help="learn from this .osr file")
    g.add_argument("--rollback", action="store_true", help="restore the previous policy")
    ap.add_argument("--replay-dirs", nargs="+", default=DEFAULT_REPLAY_DIRS)
    ap.add_argument("--songs", default=DEFAULT_SONGS)
    ap.add_argument("--policy", default="checkpoints/policy.pt", help="the policy the bot plays with")
    ap.add_argument("--human-data", default="data/replays.npz", help="your human-replay dataset (kept in the mix)")
    ap.add_argument("--player", default=None, help="only learn from replays by this player name")
    ap.add_argument("--score-only", action="store_true", help="analyse and print rewards but do not learn")
    ap.add_argument("--include-existing", action="store_true", help="in --watch, also process replays already present")
    # learning knobs (deliberately gentle)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--mix", type=float, default=0.5, help="fraction of each batch from the bot's own plays")
    ap.add_argument("--power", type=float, default=1.5, help=">1 favours clean 300s over sloppy hits")
    ap.add_argument("--buffer", type=int, default=10, help="how many recent plays to keep learning from")
    ap.add_argument("--max-drop", type=float, default=0.01, help="reject an update if sim accuracy drops by more")
    ap.add_argument("--eval-episodes", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2, help="torch CPU threads (run_bot passes 1 so play stays smooth)")
    ap.add_argument("--exit-on-stdin-eof", action="store_true",
                    help="used by run_bot.py: quit automatically when the bot (parent process) dies")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.exit_on_stdin_eof:
        threading.Thread(target=_exit_when_parent_dies, daemon=True).start()

    if a.rollback:
        rollback(a)
        return
    cfg = BotConfig()
    ctx = {"args": a, "cfg": cfg, "idx": index_songs(a.songs), "dev": torch.device(a.device),
           "rng": np.random.default_rng(0)}
    done = load_done()

    if a.replay:
        process_once(a.replay, ctx, done)
    elif a.latest:
        files = list_replays(a.replay_dirs)
        if not files:
            raise SystemExit(f"No .osr files in {a.replay_dirs}")
        process_once(max(files, key=os.path.getmtime), ctx, done)
    else:
        seen = set() if a.include_existing else set(list_replays(a.replay_dirs))
        sizes = {}
        print(f"[watch] waiting for new replays in: {a.replay_dirs}\n        (play a map with the bot; Ctrl+C to stop)")
        try:
            while True:
                for p in list_replays(a.replay_dirs):
                    if p in seen:
                        continue
                    sz = os.path.getsize(p)
                    if sz > 0 and sizes.get(p) == sz:   # unchanged since last poll -> fully written
                        seen.add(p)
                        process_once(p, ctx, done)
                    else:
                        sizes[p] = sz
                time.sleep(2.0)
        except KeyboardInterrupt:
            print("\n[watch] stopped")


if __name__ == "__main__":
    main()
