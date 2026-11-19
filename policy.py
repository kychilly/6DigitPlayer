"""The learned player.

  observation (obs.py)  ->  MLP  ->  [aim velocity (Gaussian)] + [key0 / key1 held (Bernoulli)] + value

* Aiming and tapping are BOTH learned: the Gaussian's learned std is the aim noise, the Bernoulli
  probabilities are the tap-timing distribution (so "sometimes 100s" falls out of how humans in the
  replays tapped), and which key is pressed (single vs alternate) is learned too.
* Trained in two ways (train_bc.py = copy human replays, train_rl.py / live = reward from playing).
"""
from __future__ import annotations
import os
import threading
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from config import PolicyConfig
from obs import obs_dim, x_indices, I_VX, I_VY, I_CX, I_CY, G, F, O_VALID, O_DX, O_DY, O_TTLC, O_TTLF
from model import pick_device

LOGSTD_MIN, LOGSTD_MAX = -4.5, -0.7


def mlp(i: int, hidden, act=nn.GELU):
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    return nn.Sequential(*layers), d


class ActorCritic(nn.Module):
    def __init__(self, pc: PolicyConfig):
        super().__init__()
        self.pc = pc
        n = obs_dim(pc.n_objs)
        self.actor, d = mlp(n, pc.hidden)
        self.mu = nn.Linear(d, 2)
        self.logstd = nn.Linear(d, 2)
        self.keys = nn.Linear(d, 2)
        self.critic, dc = mlp(n, (256, 128))
        self.v = nn.Linear(dc, 1)
        for m in (self.mu, self.keys):
            nn.init.orthogonal_(m.weight, 0.01); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.logstd.weight, 0.01); nn.init.constant_(self.logstd.bias, -3.0)
        nn.init.constant_(self.keys.bias, -2.0)
        nn.init.orthogonal_(self.v.weight, 1.0); nn.init.zeros_(self.v.bias)

    def forward(self, obs):
        h = self.actor(obs)
        mu = self.mu(h)
        logstd = self.logstd(h).clamp(LOGSTD_MIN, LOGSTD_MAX)
        return mu, logstd, self.keys(h), self.v(self.critic(obs)).squeeze(-1)

    # ------------------------------------------------------------------ sampling / likelihood
    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic=False, key_temp=1.0, move_scale=1.0):
        mu, logstd, kl, v = self(obs)
        std = logstd.exp()
        move = mu if deterministic else mu + std * move_scale * torch.randn_like(mu)
        p = torch.sigmoid(kl / key_temp)
        keys = (p > 0.5).float() if deterministic else torch.bernoulli(p)
        act = torch.cat([move, keys], -1)
        # log-prob under the *unscaled* policy (so PPO stays on-policy when scales are 1)
        return act, self._logp(mu, logstd, kl, act), v

    def _logp(self, mu, logstd, kl, act):
        std = logstd.exp()
        lp_move = (-0.5 * ((act[..., :2] - mu) / std) ** 2 - logstd - 0.9189385).sum(-1)
        k = act[..., 2:4]
        lp_key = -Fn.binary_cross_entropy_with_logits(kl, k, reduction="none").sum(-1)
        return lp_move + lp_key

    def evaluate(self, obs, act):
        mu, logstd, kl, v = self(obs)
        lp = self._logp(mu, logstd, kl, act)
        ent_move = (logstd + 1.4189385).sum(-1)
        p = torch.sigmoid(kl)
        ent_key = (-(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8))).sum(-1)
        return lp, ent_move + ent_key, v


# ====================================================================== BC helpers
def bc_loss(model: ActorCritic, obs, act, key_w: float = 1.0):
    """Imitation loss: Gaussian NLL on aim velocity + BCE on key state."""
    mu, logstd, kl, _ = model(obs)
    std = logstd.exp()
    nll = (0.5 * ((act[:, :2] - mu) / std) ** 2 + logstd).sum(-1).mean()
    ce = Fn.binary_cross_entropy_with_logits(kl, act[:, 2:4], reduction="none").sum(-1).mean()
    with torch.no_grad():
        mse = ((act[:, :2] - mu) ** 2).mean().item()
        key_acc = ((kl > 0).float() == act[:, 2:4]).float().mean().item()
    return nll + key_w * ce, {"nll": nll.item(), "ce": ce.item(), "mse": mse, "key_acc": key_acc}


def augment(obs: torch.Tensor, act: torch.Tensor, pc: PolicyConfig, flip=True, noise=True):
    """Mirror the playfield and add perception noise to the INPUT only (targets stay clean).
    Noise makes the policy rely on the notes, not on copying its own last velocity."""
    obs, act = obs.clone(), act.clone()
    B, n = obs.shape[0], pc.n_objs
    xs, ys = x_indices(n)
    if flip:
        fx = torch.rand(B, device=obs.device) < 0.5
        fy = torch.rand(B, device=obs.device) < 0.5
        sx = (1.0 - 2.0 * fx.float()).unsqueeze(1)
        sy = (1.0 - 2.0 * fy.float()).unsqueeze(1)
        obs[:, xs] *= sx
        obs[:, ys] *= sy
        act[:, 0] *= sx[:, 0]
        act[:, 1] *= sy[:, 0]
    if noise:
        e = torch.randn(B, 2, device=obs.device) * 4.0                      # cursor position error, px
        obs[:, I_CX] += e[:, 0] / 256.0; obs[:, I_CY] += e[:, 1] / 192.0
        vn = torch.randn(B, 2, device=obs.device) * 80.0 / 1500.0           # velocity estimate error
        obs[:, I_VX] += vn[:, 0]; obs[:, I_VY] += vn[:, 1]
        for i in range(n):
            b = G + F * i
            valid = obs[:, b + O_VALID]
            ttl = obs[:, b + O_TTLC] / 2.0
            ttl_n = ttl + torch.randn(B, device=obs.device) * pc.ttl_noise * valid
            obs[:, b + O_TTLC] = 2.0 * ttl_n.clamp(-0.4, 1.6)
            obs[:, b + O_TTLF] = torch.tanh(ttl_n / 0.04) * valid
            pn = torch.randn(B, 2, device=obs.device) * pc.pos_noise / 128.0
            obs[:, b + O_DX] += (-e[:, 0] / 128.0 + pn[:, 0]) * valid
            obs[:, b + O_DY] += (-e[:, 1] / 128.0 + pn[:, 1]) * valid
    return obs, act


# ====================================================================== save / load / runner
def save_policy(path: str, model: ActorCritic, extra: Optional[dict] = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "pc": model.pc.__dict__, "extra": extra or {}}, path)


def load_policy(path: str, pc: PolicyConfig, device) -> ActorCritic:
    ck = torch.load(path, map_location=device, weights_only=False)
    saved = ck.get("pc", {})
    for k in ("n_objs", "hidden"):                       # architecture must match the checkpoint
        if k in saved:
            setattr(pc, k, tuple(saved[k]) if k == "hidden" else saved[k])
    m = ActorCritic(pc).to(device)
    m.load_state_dict(ck["model"])
    return m


class PolicyRunner:
    """Single-observation inference wrapper for the live bot (numpy in, numpy out)."""

    def __init__(self, pc: PolicyConfig, device=None, path: Optional[str] = None):
        self.pc = pc
        self.lock = threading.Lock()          # PolicyReloader swaps weights under this lock
        self.device = device or pick_device("auto")
        path = path or pc.weights
        if os.path.exists(path):
            self.model = load_policy(path, pc, self.device)
            print(f"[policy] loaded {path} on {self.device}")
        else:
            self.model = ActorCritic(pc).to(self.device)
            print(f"[policy] WARNING: {path} not found -> untrained policy. Run train_bc.py / train_rl.py first.")
        self.model.eval()

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic=False):
        x = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with self.lock:
            a, lp, v = self.model.act(x, deterministic, self.pc.key_temp, self.pc.move_noise_scale)
        return a[0].cpu().numpy(), float(lp[0]), float(v[0])
