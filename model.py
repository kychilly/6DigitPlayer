"""CenterNet-style detector: finds hit circles, slider heads, spinners and the slider
ball, and regresses per-object (sub-pixel offset, approach progress, radius)."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


def pick_device(pref: str = "auto") -> torch.device:
    """NVIDIA (cuda) -> Intel Arc/Xe (xpu) -> cpu."""
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def device_name(dev: torch.device) -> str:
    try:
        if dev.type == "cuda":
            return torch.cuda.get_device_name(0)
        if dev.type == "xpu":
            return torch.xpu.get_device_name(0)
    except Exception:
        pass
    return "CPU - training/inference will be slow"


def cbr(i, o, k=3, s=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class ResBlock(nn.Module):
    def __init__(self, i, o, stride):
        super().__init__()
        self.c1 = cbr(i, o, 3, stride)
        self.c2 = nn.Sequential(nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o))
        self.skip = (nn.Sequential(nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o))
                     if (i != o or stride != 1) else nn.Identity())
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.c2(self.c1(x)) + self.skip(x))


class OsuNet(nn.Module):
    """Input (B,3,H,W) in [0,1]. Outputs at stride 4:
       heat: (B,num_classes,H/4,W/4) logits
       reg : (B,4,H/4,W/4) raw = [off_x, off_y, approach_progress, radius/32]"""

    def __init__(self, num_classes=4, width=24):
        super().__init__()
        w = width
        self.stem = nn.Sequential(cbr(3, w, 3, 2), cbr(w, w))
        self.s1 = nn.Sequential(ResBlock(w, 2 * w, 2), ResBlock(2 * w, 2 * w, 1))
        self.s2 = nn.Sequential(ResBlock(2 * w, 4 * w, 2), ResBlock(4 * w, 4 * w, 1))
        self.s3 = nn.Sequential(ResBlock(4 * w, 8 * w, 2), ResBlock(8 * w, 8 * w, 1))
        self.s4 = nn.Sequential(ResBlock(8 * w, 8 * w, 2), ResBlock(8 * w, 8 * w, 1))
        d = 4 * w
        self.l4, self.l3 = nn.Conv2d(8 * w, d, 1), nn.Conv2d(8 * w, d, 1)
        self.l2, self.l1 = nn.Conv2d(4 * w, d, 1), nn.Conv2d(2 * w, d, 1)
        self.sm3, self.sm2, self.sm1 = cbr(d, d), cbr(d, d), cbr(d, d)
        self.hm = nn.Sequential(cbr(d, 2 * w), nn.Conv2d(2 * w, num_classes, 1))
        self.reg = nn.Sequential(cbr(d, 2 * w), nn.Conv2d(2 * w, 4, 1))
        nn.init.constant_(self.hm[-1].bias, -2.19)

    def forward(self, x):
        x = self.stem(x)
        c1 = self.s1(x)
        c2 = self.s2(c1)
        c3 = self.s3(c2)
        c4 = self.s4(c3)
        p = self.l4(c4)
        p = self.sm3(F.interpolate(p, size=c3.shape[-2:], mode="nearest") + self.l3(c3))
        p = self.sm2(F.interpolate(p, size=c2.shape[-2:], mode="nearest") + self.l2(c2))
        p = self.sm1(F.interpolate(p, size=c1.shape[-2:], mode="nearest") + self.l1(c1))
        return self.hm(p), self.reg(p)


def reg_activate(reg):
    return torch.cat([torch.sigmoid(reg[:, 0:2]), torch.sigmoid(reg[:, 2:3]), F.softplus(reg[:, 3:4])], 1)


def focal_loss(logits, gt):
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    pos_loss = -torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = -torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * (1 - pos)
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1)


def compute_loss(heat_logits, reg_raw, gt_heat, gt_reg, gt_mask):
    heat_logits, reg_raw = heat_logits.float(), reg_raw.float()
    l_hm = focal_loss(heat_logits, gt_heat)
    pred = reg_activate(reg_raw)
    w = torch.tensor([1.0, 1.0, 2.0, 2.0], device=pred.device).view(1, 4, 1, 1)
    l_reg = (torch.abs(pred - gt_reg) * gt_mask * w).sum() / gt_mask.sum().clamp(min=1)
    return l_hm + l_reg, {"hm": l_hm.detach().item(), "reg": l_reg.detach().item()}


@dataclass
class Detection:
    cls: int
    x: float          # model-input pixels
    y: float
    progress: float   # approach ring progress: 1 = just appeared, 0 = hit time
    radius: float     # model-input pixels
    score: float


@torch.no_grad()
def decode(heat_logits, reg_raw, stride=4, thr=0.4, topk=32):
    heat = torch.sigmoid(heat_logits)
    heat = heat * (F.max_pool2d(heat, 3, 1, 1) == heat).float()  # NMS
    B, C, H, W = heat.shape
    k = min(topk, C * H * W)
    scores, inds = heat.reshape(B, -1).topk(k)
    cls = inds // (H * W)
    pix = inds % (H * W)
    ys, xs = (pix // W).float(), (pix % W).float()
    reg = reg_activate(reg_raw).reshape(B, 4, -1)
    g = reg.gather(2, pix.unsqueeze(1).expand(-1, 4, -1))
    scores, cls, xs, ys, g = [t.cpu().numpy() for t in (scores, cls, xs, ys, g)]
    out = []
    for b in range(B):
        dets = []
        for i in range(k):
            if scores[b, i] < thr:
                break  # topk is sorted
            dets.append(Detection(int(cls[b, i]), float((xs[b, i] + g[b, 0, i]) * stride),
                                  float((ys[b, i] + g[b, 1, i]) * stride),
                                  float(g[b, 2, i]), float(g[b, 3, i] * 32.0), float(scores[b, i])))
        out.append(dets)
    return out