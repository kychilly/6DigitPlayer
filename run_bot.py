"""Run the bot.   python run_bot.py --dry-run          (vision + visualizer only, no input)
                 python run_bot.py --region 0,0,1920,1080 --ar 9 --od 8 --keys "[,]"
                 python run_bot.py --policy checkpoints/policy.pt            (play with the LEARNED policy)
                 python run_bot.py --policy checkpoints/policy.pt --learn    (...and keep learning from reward)
With --policy the bot ALSO learns from each finished map's replay (see self_learn.py / learn_from_play.py);
turn that off with --no-replay-learn.
Starts DISARMED; arm with F8 (global) or SPACE (viewer). ESC anywhere = instant kill."""
import argparse
import time
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="0,0,1920,1080", help="left,top,width,height of the osu! window")
    ap.add_argument("--ar", type=float, default=9.0)
    ap.add_argument("--od", type=float, default=8.0)
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
                    help="play with the learned policy instead of the hand-written humanizer")
    ap.add_argument("--learn", action="store_true", help="with --policy: judge own taps, get reward, PPO-update after each map")
    ap.add_argument("--bc-data", default=None, help="replay dataset (npz) to keep imitating while learning live")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--save-live", default="checkpoints/policy_live.pt")
    ap.add_argument("--no-replay-learn", action="store_true",
                    help="with --policy: do NOT analyse each finished map's replay and improve the policy")
    ap.add_argument("--songs", default=None, help="osu! Songs folder (default %%LOCALAPPDATA%%\\osu!\\Songs)")
    ap.add_argument("--replay-dirs", nargs="+", default=None,
                    help="folders osu! saves replays in (default: osu!\\Replays and osu!\\Data\\r)")
    ap.add_argument("--human-data", default="data/replays.npz", help="human replay dataset kept in the learning mix")
    ap.add_argument("--player", default=None, help="only learn from replays by this in-game name")
    ap.add_argument("--replay-lr", type=float, default=2e-5, help="learning rate of the per-map replay update")
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
    if a.policy:
        from policy import PolicyRunner
        from live_learning import OnlineLearner
        torch.set_num_threads(2)
        if a.learn:
            cfg.policy.key_temp = cfg.policy.move_noise_scale = 1.0   # PPO needs on-policy sampling
        runner = PolicyRunner(cfg.policy, torch.device(a.policy_device), a.policy)
        learner = OnlineLearner(runner, cfg, lr=a.lr, bc_data=a.bc_data, save_path=a.save_live, log=shared.log) if a.learn else None
        from policy_controller import PolicyController
        controller = PolicyController(cfg, tracker, audio, backend, shared, runner, learner)
        print(f"[bot] LEARNED POLICY mode{' + live learning' if a.learn else ''}")
    else:
        controller = Controller(cfg, tracker, audio, backend, shared)
    controller.start()
    replay_learner = None
    if a.policy and a.learn:
        print("[bot] --learn (live PPO) is on, so replay learning is skipped (they would both edit the policy)")
    elif a.policy and not a.no_replay_learn:
        if a.dry_run:
            print("[bot] --dry-run plays nothing, so there are no replays to learn from")
        else:
            from self_learn import ReplayLearning
            replay_learner = ReplayLearning(a, cfg, runner, shared)
            replay_learner.start()
    KillSwitch(shared, backend, cfg.game.keys).start()
    print("[bot] ESC = emergency kill (works from any window) | F8 = arm/disarm")
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
        if replay_learner:
            replay_learner.stop()
        controller.stop(); perception.stop(); capture.stop()
        if audio:
            audio.stop()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
