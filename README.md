# 6DigitPlayer - osu! AI that learns from replays and from playing

```
screen -> CNN detector -> tracker (visible notes only, no map) -> obs
                                                                   |
        replays (.osr)  --imitation-->  POLICY NETWORK  <--PPO--  reward from playing (sim or live)
                                              |
                       aim velocity + key0/key1  ->  accel limit + tremor  ->  mouse/keys
```

Everything is a flat module in the repo root (no leading-dot / relative imports).

## What is learned
| part | how |
|---|---|
| **Reading the screen** | `train.py` - CNN detector (circles, slider heads/ball, spinners, approach progress). Unchanged. |
| **Aiming** | `policy.py` - Gaussian over cursor velocity. Mean = path, learned std = aim noise (misses happen when it is wide). Hand tremor + acceleration limit are added on top at play time. |
| **Tapping** | `policy.py` - Bernoulli key0/key1 each 1/120 s. This *is* the tap-timing distribution (normal-ish around 0 ms, occasional 100s) and it also learns single-tap vs alternate. |
| Audio beat snapping | still used to refine hit-time estimates (`audio.py`). |

## Reward (config.RewardConfig)
300 = +1.0, 100 = +0.5, 50 = +0.2, miss / off-note = 0, stray click -0.05, small bonus for tapping near 0 ms,
plus dense shaping: moving toward the next note, cursor on the note, following slider balls, spinning.

## Workflow
```
# 1. LEARN FROM REPLAYS (your own .osr files; osu!/Data/r and osu!/Songs)
python replay_data.py --replays "<osu!>/Data/r" --songs "<osu!>/Songs" --out data/replays.npz
python train_bc.py --data data/replays.npz --epochs 30           -> checkpoints/policy.pt

# 2. LEARN BY PLAYING, in the simulator (fast, ~5000 steps/s per CPU core)
python train_rl.py --init checkpoints/policy.pt --bc-data data/replays.npz --iters 300
python train_rl.py --init checkpoints/policy.pt --maps "<osu!>/Songs" --iters 500   # real maps too
                                                                  -> checkpoints/policy_rl.pt

# 3. PLAY the real game - learning from the REPLAY LIBRARY + after every map (F8 arm, ` backtick = kill)
#    replay_library/          <- put your own GOOD plays (.osr) here
#    replay_library/bot_plays <- the bot's own plays are added automatically (max 300 replays in total)
python run_bot.py --policy checkpoints/policy.pt              # studies the library, plays, learns after every map
python run_bot.py --policy checkpoints/policy.pt --no-learn   # play only, policy frozen
python rl_worker.py --study                                   # study the library now without osu!
python rl_worker.py --stats                                   # per-map log: is it improving?
python rl_worker.py --rollback                                # undo the last update
```
Check it works without osu!/replays: `python selftest.py` and `python live_selftest.py --policy <ckpt>`.

## Things to know
* Replays with DT/HT/Relax/Auto are skipped. HR/EZ are handled. Use good plays (`--min-acc`); the bot copies who it watches.
* Beatmaps are used ONLY in training/simulation. The live bot sees what the tracker sees.
* **Learning after every map (rl_worker.py):** the bot records every decision while playing; when the map ends,
  the worker finds the replay osu! saved, lines it up with the recording, re-scores every step with osu!'s real
  judgement on the real beatmap (the same reward as the simulator), runs a PPO update and writes the policy; the
  bot hot-reloads it before the next map. No replay (failed/quit map, lazer) -> falls back to the bot's own live
  judgement. Play with NoFail so every map produces a replay. Log: data/rl_log.csv, backups: checkpoints/history/.
* **Replay library (offline RL):** every replay in replay_library/ (yours first, then the bot's own plays; max 300)
  is scored with the same reward and studied with advantage-weighted regression: moments that scored better than
  expected are copied strongly, worse ones mostly ignored. A long study runs when the library changes (before the
  first map); a short refresher runs after every map. Only put GOOD plays in replay_library/ - the bot copies them.
* AR / OD are read from the osu! window title + Songs folder automatically (--ar/--od are only the fallback).
  Mods aren't visible in the title, so HR/EZ/DT maps get the wrong AR/OD (the worker warns after the map).
* Playfield mapping assumes osu!'s 80%-of-height playfield. Check the green box in the viewer; adjust `--pf-shift`.
* The detector is still trained on synthetic frames; retrain with your skin (`train.py --real-dir`) before live play.
* Only run offline / on your own client. osu! forbids automation on official servers.
