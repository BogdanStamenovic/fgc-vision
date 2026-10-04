"""Prototype stitcher v2: CLIP robot-ness filter + appearance-gated linking.

usage: stitch_v2.py KEY TRACKER [sim_floor]
Writes join montages tracks/<key>.<tracker>.v2.joins_<n>.jpg and prints chain stats.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "/home/bodas/data/fgc-vision/src")
from fgc_vision import identity as I  # noqa: E402

D = Path("/mnt/offload/fgc-vision")
os.environ.setdefault("HF_HOME", str(D / "weights/hf"))
from transformers import CLIPModel, CLIPProcessor  # noqa: E402

PROMPTS = ["a photo of a robot", "a photo of a person", "a photo of a tower with a QR code",
           "a photo of balls on a green floor"]


def main() -> None:
    key, kind = sys.argv[1], sys.argv[2]
    floor = float(sys.argv[3]) if len(sys.argv) > 3 else 0.80
    res = json.loads((D / "tracks" / f"{key}.{kind}.json").read_text())
    cap = cv2.VideoCapture(str(D / "video" / f"2025_{key}.mp4"))
    fps, s0 = res["fps"], res["start_frame"]
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().cuda()
    p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    with torch.no_grad():
        tx = p(text=PROMPTS, return_tensors="pt", padding=True).to("cuda")
        te = m.get_text_features(**tx)
        te = te if torch.is_tensor(te) else te.pooler_output
        te = te / te.norm(dim=-1, keepdim=True)

    def crop(t, b, pad=0.1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s0 + t * fps))
        ok, f = cap.read()
        w, h = b[2] - b[0], b[3] - b[1]
        return f[int(max(0, b[1] - pad * h)):int(b[3] + pad * h),
                 int(max(0, b[0] - pad * w)):int(b[2] + pad * w)]

    def emb(cs):
        x = p(images=[cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in cs],
              return_tensors="pt")["pixel_values"].cuda()
        with torch.no_grad():
            e = m.get_image_features(pixel_values=x)
        e = e if torch.is_tensor(e) else e.pooler_output
        return e / e.norm(dim=-1, keepdim=True)

    trs = I.tracklets(res)
    head, tail, robotness = {}, {}, {}
    for t in trs:
        n = len(t.t)
        hi = emb([crop(t.t[i], t.box[i]) for i in range(min(5, n))])
        ti = emb([crop(t.t[i], t.box[i]) for i in range(max(0, n - 5), n)])
        head[t.id], tail[t.id] = hi.mean(0), ti.mean(0)
        allm = torch.cat([hi, ti]).mean(0)
        allm = allm / allm.norm()
        pr = (100 * allm @ te.T).softmax(-1)
        robotness[t.id] = float(pr[0])
    kept = [t for t in trs if robotness[t.id] >= 0.5]
    print(f"{key}: {len(trs)} tracklets, {len(kept)} pass CLIP robot filter")

    # v2 stitch: same seeding/gating as identity.stitch plus appearance floor and cost
    early = sorted([t for t in kept if t.start <= 3.0], key=lambda t: -(min(t.end, 30) - t.start))
    seeds = []
    for t in early:
        if len(seeds) == 6:
            break
        if all(np.hypot(*np.subtract(t.c(0), s.c(0))) > 40 for s in seeds):
            seeds.append(t)
    chains = [I.Chain(i, "red" if s.c(0)[0] < 960 else "blue", [s]) for i, s in enumerate(seeds)]
    used = {s.id for s in seeds}
    sims = []
    for t in kept:
        if t.id in used:
            continue
        best, bc, bs = None, 1e9, 0.0
        for ch in chains:
            gap = t.start - ch.end
            if gap < -0.3 or gap > 15:
                continue
            d = float(np.hypot(*np.subtract(t.c(0), ch.last())))
            if d > 90 + 220 * max(gap, 0):
                continue
            sim = float(tail[ch.parts[-1].id] @ head[t.id])
            if sim < floor:
                continue
            cost = d / (90 + 220 * max(gap, 0)) + 4 * (1 - sim)
            if cost < bc:
                best, bc, bs = ch, cost, sim
        if best is not None:
            best.parts.append(t)
            used.add(t.id)
            sims.append(bs)
    for ch in chains:
        print(f"  chain {ch.slot} {ch.side:4} parts={len(ch.parts):2d} cov={I.coverage(ch):.2f} "
              f"end@{ch.end:.0f}s climb={I.climb(ch)['climbed']}")
    # montage
    S = 90
    items = []
    for ch in chains:
        for a, b in zip(ch.parts, ch.parts[1:]):
            items.append((ch.slot, cv2.resize(crop(a.t[-1], a.box[-1], 0.4), (S, S)),
                          cv2.resize(crop(b.t[0], b.box[0], 0.4), (S, S)), a.t[-1], b.t[0]))
    cols, per = 4, 24
    for k in range(0, len(items), per):
        chunk = items[k:k + per]
        rows = (len(chunk) + cols - 1) // cols
        img = np.full((rows * (S + 16), cols * (2 * S + 14), 3), 40, np.uint8)
        for i, (slot, c1, c2, t1, t2) in enumerate(chunk):
            r, c = divmod(i, cols)
            y, x = r * (S + 16) + 14, c * (2 * S + 14)
            img[y:y + S, x:x + S] = c1
            img[y:y + S, x + S + 2:x + 2 * S + 2] = c2
            cv2.putText(img, f"J{k + i} c{slot} {t1:.0f}->{t2:.0f}s", (x, y - 3), 0, 0.4,
                        (0, 255, 255), 1)
        cv2.imwrite(str(D / "tracks" / f"{key}.{kind}.v2.joins_{k // per}.jpg"), img)
    cov = [I.coverage(c) for c in chains]
    print(f"joins: {len(items)}, median coverage {np.median(cov):.2f}")


if __name__ == "__main__":
    main()
