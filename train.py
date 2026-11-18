"""Train the detector.  Examples:
    python train.py                              # synthetic data only
    python train.py --epochs 40 --real-dir data/real   # mix in your own labelled frames
Real-data format: data/real/foo.png + foo.json with
    {"objects":[{"cls":0,"x":0.42,"y":0.51,"progress":0.7,"radius":0.03}, ...]}
  x,y,radius are normalised by image width/height/width; cls: 0 circle,1 slider head,2 spinner,3 slider ball.
"""
import argparse
import json
import os
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from config import ModelConfig
from contextlib import nullcontext
from model import OsuNet, compute_loss, decode, pick_device, device_name
from synthetic import GTObject, SyntheticRenderer


def to_tensor(img_bgr):
    return torch.from_numpy(np.ascontiguousarray(img_bgr[:, :, ::-1])).permute(2, 0, 1).float() / 255.0


class SyntheticDataset(Dataset):
    def __init__(self, n, mc: ModelConfig, seed=None):
        self.n, self.seed = n, seed
        self.r = SyntheticRenderer(mc.input_w, mc.input_h, mc.stride, mc.num_classes)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.seed is None else self.seed + i)
        img, objs = self.r.render(rng)
        h, r, m = self.r.build_targets(objs)
        return to_tensor(img), torch.from_numpy(h), torch.from_numpy(r), torch.from_numpy(m)


class RealDataset(Dataset):
    def __init__(self, folder, mc: ModelConfig, repeat=20):
        self.items = [(p, p.with_suffix(".json")) for p in sorted(Path(folder).glob("*.png")) if p.with_suffix(".json").exists()]
        self.mc, self.repeat = mc, repeat
        self.r = SyntheticRenderer(mc.input_w, mc.input_h, mc.stride, mc.num_classes)
        print(f"[data] {len(self.items)} real labelled frames")

    def __len__(self):
        return len(self.items) * self.repeat

    def __getitem__(self, i):
        png, js = self.items[i % len(self.items)]
        img = cv2.resize(cv2.imread(str(png)), (self.mc.input_w, self.mc.input_h), interpolation=cv2.INTER_AREA)
        objs = [GTObject(o["cls"], o["x"] * self.mc.input_w, o["y"] * self.mc.input_h, o["progress"], o["radius"] * self.mc.input_w)
                for o in json.load(open(js))["objects"]]
        img = np.clip(img.astype(np.float32) * np.random.uniform(0.85, 1.15), 0, 255).astype(np.uint8)  # light jitter
        h, r, m = self.r.build_targets(objs)
        return to_tensor(img), torch.from_numpy(h), torch.from_numpy(r), torch.from_numpy(m)


@torch.no_grad()
def evaluate(model, loader, dev, mc):
    model.eval()
    tp = fp = fn = 0
    loss_sum, n = 0.0, 0
    perr = []
    for img, h, r, m in loader:
        img, h, r, m = img.to(dev), h.to(dev), r.to(dev), m.to(dev)
        hl, rg = model(img)
        loss_sum += float(compute_loss(hl, rg, h, r, m)[0]); n += 1
        dets = decode(hl, rg, mc.stride, mc.score_threshold, mc.max_dets)
        for b, ds in enumerate(dets):
            ys, xs = torch.nonzero(m[b, 0], as_tuple=True)
            gts = [(int(h[b, :, y, x].argmax()), float(r[b, 0, y, x] + x) * mc.stride, float(r[b, 1, y, x] + y) * mc.stride,
                    float(r[b, 2, y, x]), float(r[b, 3, y, x]) * 32) for y, x in zip(ys.tolist(), xs.tolist())]
            used = set()
            for d in ds:
                hit = None
                for gi, g in enumerate(gts):
                    if gi not in used and g[0] == d.cls and np.hypot(d.x - g[1], d.y - g[2]) < max(4, 0.6 * g[4]):
                        hit = gi; break
                if hit is None:
                    fp += 1
                else:
                    used.add(hit); tp += 1; perr.append(abs(d.progress - gts[hit][3]))
            fn += len(gts) - len(used)
    p, r_ = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return loss_sum / max(1, n), p, r_, (float(np.mean(perr)) if perr else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=300, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--real-dir", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out", default="checkpoints/detector.pt")
    a = ap.parse_args()

    mc = ModelConfig()
    dev = pick_device()
    use_amp = dev.type == "cuda"
    print(f"[train] device: {dev} ({device_name(dev)})", flush=True)
    model = OsuNet(mc.num_classes).to(dev)
    if a.resume:
        model.load_state_dict(torch.load(a.resume, map_location=dev)["model"])

    train_sets = [SyntheticDataset(a.steps * a.batch, mc)]
    if a.real_dir:
        train_sets.append(RealDataset(a.real_dir, mc))
    train_dl = DataLoader(ConcatDataset(train_sets), batch_size=a.batch, shuffle=True, num_workers=a.workers,
                          drop_last=True, persistent_workers=a.workers > 0)
    val_dl = DataLoader(SyntheticDataset(256, mc, seed=12345), batch_size=32, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(train_dl))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    best = 1e9
    for ep in range(a.epochs):
        model.train()
        t0, run = time.time(), 0.0
        print(f"epoch {ep + 1}/{a.epochs} started ({len(train_dl)} batches)...", flush=True)
        for bi, (img, h, r, m) in enumerate(train_dl):
            img, h, r, m = img.to(dev), h.to(dev), r.to(dev), m.to(dev)
            with (torch.autocast(device_type='cuda') if use_amp else nullcontext()):
                hl, rg = model(img)
            loss, parts = compute_loss(hl, rg, h, r, m)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.detach().item()
            if (bi + 1) % 25 == 0:
                print(f"  batch {bi + 1}/{len(train_dl)} loss {loss.detach().item():.3f} ({time.time() - t0:.0f}s)", flush=True)
        vl, p, rc, perr = evaluate(model, val_dl, dev, mc)
        print(f"epoch {ep + 1:3d}/{a.epochs} train {run / len(train_dl):.3f} val {vl:.3f} "
              f"P {p:.3f} R {rc:.3f} progress-err {perr:.3f} ({time.time() - t0:.0f}s)")
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict()}, a.out)
            print(f"  saved {a.out}")


if __name__ == "__main__":
    main()
