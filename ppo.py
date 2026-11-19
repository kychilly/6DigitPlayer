"""PPO for the osu! policy. Used by the simulator trainer (train_rl.py) AND by live learning
(policy_controller.py -> OnlineLearner), so both improve the same network from the same reward.

BC regulariser: while doing RL we keep mixing in a little imitation loss on the human replays, so
the policy keeps its human-like movement while it learns to be more accurate (it can't "forget"
how to move like a person just to chase reward)."""
from __future__ import annotations
import numpy as np
import torch

from config import BotConfig
from policy import ActorCritic, bc_loss, augment


def compute_gae(rew, val, done, last_val, gamma, lam):
    """rew/val/done: (T, N) float arrays; last_val: (N,)."""
    T = len(rew)
    adv = np.zeros_like(rew)
    last = np.zeros_like(last_val)
    for t in reversed(range(T)):
        nv = last_val if t == T - 1 else val[t + 1]
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nv * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + val


class PPOLearner:
    def __init__(self, model: ActorCritic, cfg: BotConfig, device, lr=3e-5, bc_data=None, bc_coef=0.5,
                 clip=0.2, epochs=4, minibatch=2048, ent_coef=0.0, vf_coef=0.5, lam=0.95, critic_warmup=0,
                 target_kl=None):
        self.m, self.cfg, self.dev = model, cfg, device
        self.gamma, self.lam = cfg.reward.gamma, lam
        self.clip, self.epochs, self.mb, self.ent, self.vf = clip, epochs, minibatch, ent_coef, vf_coef
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.bc_coef, self.warm, self.iters = bc_coef, critic_warmup, 0
        self.target_kl = target_kl          # stop an update early once the policy has moved this far
        self.bc = None
        if bc_data is not None and bc_coef > 0:
            d = np.load(bc_data)
            self.bc = (torch.from_numpy(d["obs"]).to(device), torch.from_numpy(d["act"]).to(device))

    def update(self, obs, act, logp, val, rew, done, last_val) -> dict:
        """All arrays are (T, N, ...) numpy; returns training statistics."""
        T, N = rew.shape
        adv, ret = compute_gae(rew, val, done, last_val, self.gamma, self.lam)
        dev = self.dev
        f = lambda x: torch.from_numpy(np.ascontiguousarray(x.reshape(T * N, *x.shape[2:]))).float().to(dev)
        o, a, lp0, ad, rt = f(obs), f(act), f(logp), f(adv), f(ret)
        ad = (ad - ad.mean()) / (ad.std() + 1e-8)
        n = T * N
        actor_on = self.iters >= self.warm
        stats = {"pg": 0.0, "vf": 0.0, "kl": 0.0, "bc": 0.0, "ent": 0.0}
        cnt = 0
        self.m.train()
        stopped = False
        for _ in range(self.epochs):
            if stopped:
                break
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, self.mb):
                i = perm[s:s + self.mb]
                lp, ent, v = self.m.evaluate(o[i], a[i])
                ratio = (lp - lp0[i]).exp()
                pg = -torch.min(ratio * ad[i], ratio.clamp(1 - self.clip, 1 + self.clip) * ad[i]).mean()
                vf = 0.5 * (v - rt[i]).pow(2).mean()
                loss = self.vf * vf
                if actor_on:
                    loss = loss + pg - self.ent * ent.mean()
                bcl = torch.zeros((), device=dev)
                if self.bc is not None and actor_on:
                    j = torch.randint(0, len(self.bc[0]), (self.mb,), device=dev)
                    bo, ba = augment(self.bc[0][j], self.bc[1][j], self.cfg.policy)
                    bcl, _ = bc_loss(self.m, bo, ba)
                    loss = loss + self.bc_coef * bcl * 0.1
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.m.parameters(), 0.5)
                self.opt.step()
                stats["pg"] += pg.item(); stats["vf"] += vf.item(); stats["bc"] += float(bcl.detach())
                kl = (lp0[i] - lp).mean().item()
                stats["kl"] += kl; stats["ent"] += ent.mean().item()
                cnt += 1
                if self.target_kl is not None and actor_on and kl > 1.5 * self.target_kl:
                    stopped = True          # policy moved enough for one batch of data
                    break
        self.iters += 1
        self.m.eval()
        out = {k: v / max(cnt, 1) for k, v in stats.items()}
        out["actor_on"], out["minibatches"], out["early_stop"] = actor_on, cnt, stopped
        return out

    # ------------------------------------------------------------------ persistence (optimizer + counters)
    def state_dict(self) -> dict:
        return {"opt": self.opt.state_dict(), "iters": self.iters}

    def load_state_dict(self, st: dict):
        self.iters = int(st.get("iters", 0))
        try:
            self.opt.load_state_dict(st["opt"])
        except Exception:                       # architecture changed -> fresh optimizer is fine
            pass
