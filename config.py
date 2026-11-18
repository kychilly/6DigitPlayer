from __future__ import annotations
from dataclasses import dataclass, field

# Detector classes
CIRCLE, SLIDER_HEAD, SPINNER, SLIDER_BALL = 0, 1, 2, 3
CLASS_NAMES = ["circle", "slider_head", "spinner", "slider_ball"]


def ar_to_preempt(ar: float) -> float:
    """Seconds an object is visible before it must be hit (osu! formula)."""
    if ar < 5:
        ms = 1200 + 600 * (5 - ar) / 5
    elif ar == 5:
        ms = 1200
    else:
        ms = 1200 - 750 * (ar - 5) / 5
    return ms / 1000.0


@dataclass
class CaptureConfig:
    # Screen rectangle that contains the osu! window (pixels, virtual-desktop coords)
    left: int = 0
    top: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 120
    display_latency: float = 0.025  # seconds between game state and captured pixels


@dataclass
class ModelConfig:
    input_w: int = 384
    input_h: int = 224
    stride: int = 4
    num_classes: int = 4
    weights: str = "checkpoints/detector.pt"
    score_threshold: float = 0.40
    max_dets: int = 32
    device: str = "auto"  # auto | cpu | cuda


@dataclass
class GameConfig:
    ar: float = 9.0
    od: float = 8.0
    keys: tuple = ("[", "]")   # must match your osu! key bindings
    stream_gap: float = 0.20  # seconds; gaps below this => alternate taps


@dataclass
class TrackerConfig:
    min_hits: int = 2          # detections before an object is trusted
    forget_after: float = 0.25  # drop objects unseen for this long
    ema_pos: float = 0.5
    ema_time: float = 0.3
    suppress_time: float = 0.35  # ignore "ghost" of a circle we just clicked
    # "is this a real note?" physics check: a real circle's estimated hit time stays put as the
    # clock advances; a static thing (menu logo) has constant ring progress, so its estimate drifts.
    verify_age: float = 0.2      # seconds of tracking before judging
    verify_tol: float = 0.09     # max allowed drift of the hit-time estimate (s)
    max_radius_frac: float = 0.16  # circle radius limits as a fraction of frame height
    min_radius_frac: float = 0.01


@dataclass
class HumanConfig:
    # ---- aiming ----
    tremor_amp_px: float = 0.7      # 8-12 Hz physiological tremor (at 1080p)
    tremor_drift_px: float = 1.5    # slow drift
    aim_sigma_frac: float = 0.10    # endpoint scatter, fraction of circle radius
    aim_speed_gain: float = 0.06    # extra scatter at high speed
    miss_base: float = 0.01         # chance of an outright aim miss
    miss_gain: float = 0.06         # extra miss chance at high speed
    curve_mean: float = 0.05        # path curvature as fraction of distance
    curve_std: float = 0.035
    fitts_a: float = 0.045
    fitts_b: float = 0.085
    speed_factor: float = 1.0       # >1 = slower hands
    min_move_time: float = 0.03
    lead_mean: float = 0.012        # arrive on circle slightly before tapping
    lead_std: float = 0.008
    # ---- tapping ----
    tap_mu: float = 0.004           # mean lateness (s)
    p100: float = 0.07              # chance of a "100"-quality tap
    p50: float = 0.012              # chance of a "50"-quality tap
    stream_p100_mult: float = 1.6   # sloppier in streams
    hold_mean: float = 0.045
    hold_std: float = 0.010
    latency_comp: float = 0.0       # extra input-latency compensation (s)
    # ---- audio assist ----
    audio_window: float = 0.040
    audio_weight: float = 0.5
    # ---- sliders / spinners ----
    follow_rate: float = 22.0
    follow_noise_px: float = 1.2
    spin_omega_mean: float = 32.0   # rad/s (~5 rev/s)
    spin_omega_std: float = 4.0


@dataclass
class AudioConfig:
    enabled: bool = True
    samplerate: int = 44100
    hop: int = 512
    n_fft: int = 1024
    offset: float = 0.0  # seconds; calibrate audio-vs-visual latency


@dataclass
class GateConfig:
    """When the bot may touch your mouse/keyboard."""
    require_focus: bool = True     # osu! must be the foreground window
    process_prefix: str = "osu"    # foreground process name must start with this (osu!.exe)
    engage_after: float = 0.0      # extra wait after a verified note appears
    title_gate: bool = False       # also require a "playing" window title (see --title-gate)
    idle_release: float = 4.0      # give control back after this long with no objects visible


# ======================================================================================
# Learned-policy settings (replay imitation + reinforcement learning)
# ======================================================================================
@dataclass
class PlayfieldConfig:
    """Maps osu! playfield coordinates (512x384) <-> capture-frame pixels.
    osu! draws the playfield at 80% of screen height. If circles look shifted in the viewer's
    green playfield rectangle, tweak y_shift (in osu!px) until the rectangle hugs the real field."""
    scale: float = 0.0       # 0 = auto (frame_height / 480)
    x0: float = 0.0          # 0 = auto (centred)
    y0: float = 0.0          # 0 = auto (centred + y_shift)
    y_shift: float = 0.0


@dataclass
class RewardConfig:
    # judgement rewards (what the user asked for: hit = full, 100/50 = less, off-note = none)
    r300: float = 1.0
    r100: float = 0.5
    r50: float = 0.2
    miss: float = 0.0
    stray_click: float = -0.05      # key press that hit nothing
    timing_bonus: float = 0.15      # extra for a tap near 0 ms (gaussian, sigma below)
    timing_sigma: float = 0.012
    # dense shaping
    on_target: float = 0.01         # per step: cursor is on the note that is due
    approach_shaping: float = 0.5   # potential-based: get close to the next note
    slider_follow: float = 0.02     # per step following the slider ball with a key down
    spin_per_rev: float = 0.25      # per spinner revolution (capped per spinner by the required amount)
    idle_key_penalty: float = 0.002 # per step a key is held with nothing to hit
    gamma: float = 0.995


@dataclass
class PolicyConfig:
    hz: int = 120                   # decisions per second (sim, dataset and live must all agree)
    n_objs: int = 6                 # how many upcoming notes the policy sees
    hidden: tuple = (256, 256, 128)
    max_speed: float = 1500.0       # osu!px/s represented by an action of 1.0
    max_accel: float = 400000.0     # osu!px/s^2 hand-acceleration limit
    weights: str = "checkpoints/policy.pt"
    key_temp: float = 1.0           # >1 = sloppier / more random taps at play time
    move_noise_scale: float = 1.0   # scale on the learned aim noise at play time
    # perception noise injected in sim / BC so the policy tolerates real detector noise
    ttl_noise: float = 0.008        # s
    pos_noise: float = 3.0          # osu!px


@dataclass
class BotConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    game: GameConfig = field(default_factory=GameConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    human: HumanConfig = field(default_factory=HumanConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    playfield: PlayfieldConfig = field(default_factory=PlayfieldConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    control_hz: int = 1000
