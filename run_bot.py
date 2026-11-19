"""Run the bot.
    python run_bot.py --dry-run                                  (vision + visualizer only, no input)
    python run_bot.py                                            (hand-written humanizer, no learning)
    python run_bot.py --policy                                   (LEARNED policy + reinforcement learning after every map)
    python run_bot.py --policy --no-learn                        (learned policy, frozen - no updates)
    python run_bot.py --policy checkpoints/policy.pt --region 0,0,1920,1080 --keys "[,]"

With --policy, the bot learns from a REPLAY LIBRARY (replay_library/: your own good plays + its own past plays,
max 300): when you start it, it first studies the library if it changed; after every map osu! saves the replay ->
rl_worker.py scores it with osu!'s real rules (300/100/50/miss, stray clicks, aim, slider follow, spinners) ->
PPO update on that play + a refresher over the library -> the bot loads the improved policy before the next map.
See rl_worker.py. Progress: python rl_worker.py --stats

AR / OD are read automatically from the osu! window title (--ar / --od are only the fallback).
Starts DISARMED; arm with F8 (global) or SPACE (viewer). ` (backtick) anywhere = instant kill."""
import argparse
import os
import time
from pathlib import Path

from config import BotConfig
from capture import ScreenCapture
from perception import Detector, PerceptionThread
from tracker import ObjectTracker
from audio import AudioListener
from inputs import make_backend
from controller import Controller
import torch
from state import SharedState
from visualizer import Visualizer
from killswitch import KillSwitch

DEFAULT_SONGS = str(Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "osu!" / "Songs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="0,0,1920,1080", help="left,top,width,height of the osu! window")
    ap.add_argument("--ar", type=float, default=9.0, help="fallback AR when the map can't be auto-detected")
    ap.add_argument("--od", type=float, default=8.0, help="fallback OD when the map can't be auto-detected")
    ap.add_argument("--no-auto-diff", action="store_true", help="don't read AR/OD from the osu! window title")
    ap.add_argument("--keys", default="[,]", help="two tap keys, comma-separated (default [ and ]); must match osu! bindings")
    ap.add_argument("--weights", default="checkpoints/detector.pt")
    ap.add_argument("--dry-run", action="store_true", help="never send mouse/keyboard input")
    ap.add_argument("--armed", action="store_true", help="start armed (needed with --no-viz)")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--no-focus-check", action="store_true", help="don't require osu! to be the focused window")
    ap.add_argument("--title-gate", action="store_true",
                    help="also require the osu! window title to show a map (check the 'focus:' line in the viewer first)")
    ap.add_argument("--idle-release", type=float, default=4.0, help="seconds without objects before giving control back")
    ap.add_argument("--policy", nargs="?", const="checkpoints/policy.pt", default=None,
                    help="play with the learned policy (and learn after every map) instead of the hand-written humanizer")
    # ---- reinforcement learning after every map
    ap.add_argument("--no-learn", action="store_true", help="with --policy: play only, never update the policy")
    ap.add_argument("--no-replay-learn", dest="no_learn", action="store_true", help=argparse.SUPPRESS)  # old name
    ap.add_argument("--learn", action="store_true", help=argparse.SUPPRESS)   # old flag; learning is now the default
    ap.add_argument("--rl-lr", type=float, default=1e-4, help="PPO learning rate per map")
    ap.add_argument("--bc-coef", type=float, default=0.5,
                    help="how strongly RL keeps imitating your human replays (keeps movement human; 0 = pure RL)")
    ap.add_argument("--no-live-fallback", action="store_true",
                    help="skip maps with no saved replay instead of learning from the bot's own judgement")
    ap.add_argument("--songs", default=None, help="osu! Songs folder (default %%LOCALAPPDATA%%\\osu!\\Songs)")
    ap.add_argument("--replay-dirs", nargs="+", default=None,
                    help="folders osu! saves replays in (default: osu!\\Replays and osu!\\Data\\r)")
    ap.add_argument("--max-replays", type=int, default=300,
                    help="replay library size: your plays in replay_library/ + the bot's own plays (newest first)")
    ap.add_argument("--human-data", default="", help=argparse.SUPPRESS)   # old npz dataset; the replay library replaces it
    ap.add_argument("--policy-device", default="cpu", help="cpu (recommended for the tiny net) | cuda | xpu")
    ap.add_argument("--pf-shift", type=float, default=0.0, help="shift the playfield down by N osu!px if the green box is off")
    ap.add_argument("--display-latency", type=float, default=0.025)
    ap.add_argument("--audio-offset", type=float, default=0.0)
    a = ap.parse_args()

    cfg = BotConfig()
    l, t, w, h = (int(v) for v in a.region.split(","))
    cfg.capture.left, cfg.capture.top, cfg.capture.width, cfg.capture.height = l, t, w, h
    cfg.capture.display_latency = a.display_latency
    cfg.game.ar, cfg.game.od, cfg.game.keys = a.ar, a.od, tuple(a.keys.split(","))
    cfg.model.weights = a.weights
    cfg.playfield.y_shift = a.pf_shift
    cfg.audio.offset = a.audio_offset
    cfg.gate.require_focus = not a.no_focus_check
    cfg.gate.idle_release = a.idle_release
    cfg.gate.title_gate = a.title_gate

    shared = SharedState()
    shared.armed = a.armed and not a.dry_run
    capture = ScreenCapture(cfg.capture); capture.start()
    detector = Detector(cfg.model)
    tracker = ObjectTracker(cfg.game, cfg.tracker)
    audio = None
    if cfg.audio.enabled and not a.no_audio:
        audio = AudioListener(cfg.audio); audio.start()
    backend = make_backend(a.dry_run, (l + w // 2, t + h // 2))
    perception = PerceptionThread(capture, detector, tracker, shared, cfg.model, cfg.capture); perception.start()
    diff_watch = None
    if not a.no_auto_diff:
        from diff_detect import DiffWatcher
        diff_watch = DiffWatcher(a.songs or DEFAULT_SONGS, shared, tracker, log=print)
        diff_watch.start()

    learning = bool(a.policy) and not a.no_learn and not a.dry_run
    if a.policy:
        from policy import PolicyRunner
        from live_learning import EpisodeRecorder
        from policy_controller import PolicyController
        torch.set_num_threads(2)
        if learning:
            cfg.policy.key_temp = cfg.policy.move_noise_scale = 1.0   # PPO needs the policy's own sampling
        runner = PolicyRunner(cfg.policy, torch.device(a.policy_device), a.policy)
        recorder = EpisodeRecorder(cfg, log=shared.log) if learning else None
        controller = PolicyController(cfg, tracker, audio, backend, shared, runner, recorder)
        print(f"[bot] LEARNED POLICY mode{' + reinforcement learning after every map' if learning else ' (frozen)'}")
    else:
        controller = Controller(cfg, tracker, audio, backend, shared)
        print("[bot] hand-written humanizer mode (no learning). Use --policy for the learning AI.")
    controller.start()
    rl = None
    if a.policy and a.dry_run and not a.no_learn:
        print("[bot] --dry-run plays nothing, so there is nothing to learn from")
    if learning:
        from self_learn import ReplayLearning
        rl = ReplayLearning(a, cfg, runner, shared)
        rl.start()
    KillSwitch(shared, backend, cfg.game.keys).start()
    print("[bot] ` (backtick) = emergency kill (works from any window) | F8 = arm/disarm")
    print("[bot] Bot only takes the mouse while osu! is focused AND a map's objects are on screen.")

    try:
        if a.no_viz:
            while not shared.quit:
                time.sleep(0.2)
        else:
            Visualizer(cfg, tracker, audio, shared).run()
    except KeyboardInterrupt:
        pass
    finally:
        if rl:
            rl.stop()
        if diff_watch:
            diff_watch.stop()
        controller.stop(); perception.stop(); capture.stop()
        if audio:
            audio.stop()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
