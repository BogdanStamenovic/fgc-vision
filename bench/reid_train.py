"""Train a re-ID projection head on frozen DINOv2-base features, supervised by tracklets.

Positives: two crops of the same tracklet. Negatives: crops of tracklets in the same match
that overlap it in time (two boxes on screen at once are two different robots).
Non-overlapping tracklets are neither (they might be the same robot). t2-1 is held out.

Evaluations:
  1. t2-1 fragment -> start-gallery classification on labels/frag_ids_t2-1.json
  2. cross-match consensus gap (real vs shuffled-team objective) on pre/galleries.npz crops
usage: reid_train.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

D = Path("/mnt/offload/fgc-vision")
os.environ.setdefault("HF_HOME", str(D / "weights/hf"))
TRAIN = ["t2-30", "t2-2", "t2-16", "t2-304", "t2-50", "t4-3", "t2-308", "t2-361"]
HOLD = "t2-1"


def dino():
    from transformers import AutoImageProcessor, AutoModel
    m = AutoModel.from_pretrained("facebook/dinov2-base", dtype=torch.float16).eval().cuda()
    p = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    def f(imgs):
        out = []
        for i in range(0, len(imgs), 64):
            x = p(images=list(imgs[i:i + 64]), return_tensors="pt")["pixel_values"].cuda().half()
            with torch.no_grad():
                o = m(pixel_values=x)
            out.append(torch.cat([o.last_hidden_state[:, 0], o.last_hidden_state[:, 1:].mean(1)],
                                 -1).float().cpu())
        return torch.cat(out)
    return f


def load(key, f):
    z = np.load(D / "reid" / f"{key}.npz")
    meta = json.loads(str(z["meta"]))
    fr = z["frag"][..., ::-1].copy()
    gal = z["gal"][..., ::-1].copy()
    n = len(fr)
    F = f(fr.reshape(-1, *fr.shape[2:])).reshape(n, fr.shape[1], -1)
    G = f(gal.reshape(-1, *gal.shape[2:])).reshape(len(gal), gal.shape[1], -1)
    return meta, F, G


class Head(nn.Module):
    def __init__(self, d=1536):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, 128))

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=-1)


def main() -> None:
    torch.manual_seed(0)
    f = dino()
    data = {k: load(k, f) for k in TRAIN + [HOLD]}
    head = Head().cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    tau = 0.1
    for step in range(1500):
        key = TRAIN[step % len(TRAIN)]
        meta, F, _ = data[key]
        fr = meta["frag"]
        n = len(fr)
        idx = np.random.choice(n, size=min(48, n), replace=False)
        a = np.random.randint(0, F.shape[1], size=len(idx))
        b = (a + np.random.randint(1, F.shape[1], size=len(idx))) % F.shape[1]
        xa = head(F[idx, a].cuda())
        xb = head(F[idx, b].cuda())
        sim = xa @ xb.T / tau
        ov = torch.tensor([[fr[i]["t0"] < fr[j]["t1"] and fr[j]["t0"] < fr[i]["t1"] for j in idx]
                           for i in idx], device="cuda")
        eye = torch.eye(len(idx), dtype=torch.bool, device="cuda")
        valid = ov | eye           # only concurrent tracklets act as negatives
        sim = sim.masked_fill(~valid, -1e4)
        loss = nn.functional.cross_entropy(sim, torch.arange(len(idx), device="cuda"))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 300 == 0:
            print(f"step {step} loss {loss.item():.3f}", flush=True)
    head.eval()
    torch.save(head.state_dict(), D / "weights" / "reid_head.pt")

    # eval 1: held-out t2-1 classification
    lab = json.loads(Path("/home/bodas/data/fgc-vision/labels/frag_ids_t2-1.json").read_text())
    truth = {i: int(k[1]) for k, v in lab.items() if k.startswith("G") for i in v}
    meta, F, G = data[HOLD]
    for name, proj in (("frozen", lambda x: nn.functional.normalize(x, dim=-1)),
                       ("trained", lambda x: head(x.cuda()).cpu())):
        with torch.no_grad():
            fm = nn.functional.normalize(proj(F).mean(1), dim=-1)
            gm = nn.functional.normalize(proj(G).mean(1), dim=-1)
        pred = (fm @ gm.T).argmax(1).numpy()
        acc = np.mean([pred[i] == t for i, t in truth.items()])
        print(f"t2-1 held-out fragment accuracy ({name}): {acc:.2f} on {len(truth)}")

    # eval 2: consensus gap on pre galleries, using the 4 stored crops per row
    z = np.load(D / "pre" / "galleries.npz")
    crops = z["crops"][..., ::-1].copy()
    Fg = f(crops.reshape(-1, *crops.shape[2:])).reshape(len(crops), crops.shape[1], -1)
    with torch.no_grad():
        Et = head(Fg.cuda()).mean(1).cpu().numpy()
    Ef = nn.functional.normalize(Fg, dim=-1).mean(1).numpy()
    np.savez(D / "pre" / "galleries_reid.npz", meta=z["meta"], crops=z["crops"],
             reid=Et, dinov2_4crop=Ef)
    print("wrote pre/galleries_reid.npz (embeddings 'reid' and 'dinov2_4crop')")


if __name__ == "__main__":
    sys.exit(main())
