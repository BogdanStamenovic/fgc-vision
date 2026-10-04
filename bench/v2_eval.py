"""v2 evaluation on t2-1 against the time-sampled hand ground truth (labels/gtp_t2-1.json).

1. Linking: pixel stitcher (v1) vs field-metre Kalman stitcher. Each chain carries the
   identity of its seed (start robot); score = share of identified GT samples whose box
   belongs to a chain of the right robot (chains include only what they linked).
2. Fragment classification (DINOv2-base vs start galleries, alliance from start side not
   used because LEDs are invisible) and the human-tap simulation: taps go to the
   fragments with the smallest top-2 margin first (or a random order as control); a tap on
   a fragment returns my label at one of its GT samples ('?' = the human could not tell,
   the request stays open and costs the tap). A tapped fragment takes the answer.
   Reports accuracy over identified GT samples vs taps.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/bodas/data/fgc-vision/src")
sys.path.insert(0, "/home/bodas/data/fgc-vision/bench")
from fgc_vision import identity as I  # noqa: E402
from gt_session import fragments  # noqa: E402

D = Path("/mnt/offload/fgc-vision")
KEY = "t2-1"
GT = json.loads(Path(f"/home/bodas/data/fgc-vision/labels/gtp_{KEY}.json").read_text())
START = GT["start"]
SEED_LETTER = {v: k for k, v in START.items()}
SEED_LETTER[89] = "E"   # E's start had two fragments


def samples():
    out = []   # (t, fragment id, truth)
    for t, pts in GT["points"].items():
        for fid, lab in pts.items():
            if lab not in ("N", "D2", "?"):
                out.append((float(t), int(fid), lab))
    return out


class Cam:
    def __init__(self, d):
        from fgc_vision import calibrate as C
        self.p = np.r_[d["k1"], d["H"][:8]]
        self.C = C

    def to_field(self, uv):
        return self.C.apply(self.p, np.asarray(uv, float).reshape(-1, 2))


def chain_identity(chains):
    owner = {}
    for ch in chains:
        letter = SEED_LETTER.get(ch.parts[0].id, "?")
        for p in ch.parts:
            owner[p.id] = letter
    return owner


def linking(res, trs, cam):
    S = samples()
    out = {}
    for name, chains in (("pixel (v1)", I.stitch(trs)),
                         ("metres+Kalman", I.stitch_metric(trs, cam))):
        own = chain_identity(chains)
        covered = [s for s in S if s[1] in own]
        right = sum(own[s[1]] == s[2] for s in covered)
        joins = sum(len(c.parts) - 1 for c in chains)
        out[name] = {"joins": joins, "gt_samples": len(S), "covered": len(covered),
                     "correct": right,
                     "acc_on_covered": round(right / max(1, len(covered)), 3),
                     "acc_on_all": round(right / len(S), 3)}
    return out


def classify(res, trs):
    import torch
    os.environ.setdefault("HF_HOME", str(D / "weights/hf"))
    from transformers import AutoImageProcessor, AutoModel
    import cv2
    m = AutoModel.from_pretrained("facebook/dinov2-base", dtype=torch.float16).eval().cuda()
    p = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    cap = cv2.VideoCapture(str(D / "video" / f"2025_{KEY}.mp4"))
    s0, fps = res["start_frame"], res["fps"]

    def emb(t, box):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(s0 + t * fps)))
        f = cap.read()[1]
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        c = f[int(max(0, y0 - .12 * h)):int(y1 + .12 * h), int(max(0, x0 - .12 * w)):int(x1 + .12 * w)]
        c = cv2.resize(c, (128, 128), interpolation=cv2.INTER_CUBIC)[..., ::-1].copy()
        return c

    def embed(crops):
        x = p(images=crops, return_tensors="pt")["pixel_values"].cuda().half()
        with torch.no_grad():
            o = m(pixel_values=x)
        e = torch.cat([o.last_hidden_state[:, 0], o.last_hidden_state[:, 1:].mean(1)], -1).float()
        e = torch.nn.functional.normalize(e, dim=-1).mean(0)
        return (e / e.norm()).cpu().numpy()
    by = {t.id: t for t in trs}
    G = {}
    for letter, fid in START.items():
        t = by[fid]
        idx = [i for i, tt in enumerate(t.t) if tt < -0.5]
        idx = idx[:: max(1, len(idx) // 10)][:10]
        G[letter] = embed([emb(t.t[i], t.box[i]) for i in idx])
    F = {}
    for t in trs:
        idx = np.linspace(0, len(t.t) - 1, min(6, len(t.t))).astype(int)
        F[t.id] = embed([emb(t.t[i], t.box[i]) for i in idx])
    letters = sorted(G)
    Gm = np.stack([G[k] for k in letters])
    scores = {fid: f @ Gm.T for fid, f in F.items()}
    return letters, scores


def taps(letters, scores, order="margin", seed=0):
    S = samples()
    frag_samples = {}
    for t, fid, lab in S:
        frag_samples.setdefault(fid, []).append((t, lab))
    # what a human answers when asked about a fragment: my label at a sampled time of it,
    # including '?' ones (the human cannot tell either)
    allpts = {}
    for t, pts in GT["points"].items():
        for fid, lab in pts.items():
            allpts.setdefault(int(fid), []).append(lab)
    pred = {fid: letters[int(np.argmax(s))] for fid, s in scores.items()}
    for fid, letter in SEED_LETTER.items():
        pred[fid] = letter          # start anchors are free: seeded by position
    rng = random.Random(seed)
    cand = [fid for fid in scores if fid in allpts and fid not in SEED_LETTER]
    if order == "margin":
        cand.sort(key=lambda f: -np.diff(np.sort(scores[f])[-2:])[0] * -1)
        cand.sort(key=lambda f: float(np.sort(scores[f])[-1] - np.sort(scores[f])[-2]))
    else:
        rng.shuffle(cand)

    def acc():
        return np.mean([pred.get(fid) == lab for _, fid, lab in S])
    curve = [(0, round(acc(), 3))]
    n = 0
    for fid in cand:
        n += 1
        ans = rng.choice(allpts[fid])
        if ans in letters:
            pred[fid] = ans
        curve.append((n, round(acc(), 3)))
    return curve


def main() -> None:
    res = json.loads((D / "tracks" / f"{KEY}.pre.json").read_text())
    trs = fragments(res)
    cam = Cam(json.loads((D / "calib" / "2025_d1f1.json").read_text()))
    print("samples with known identity:", len(samples()))
    print(json.dumps(linking(res, I.tracklets(res, min_n=5), cam), indent=1))
    letters, scores = classify(res, trs)
    S = samples()
    raw = np.mean([letters[int(np.argmax(scores[fid]))] == lab for _, fid, lab in S if fid in scores])
    print("DINOv2 per-fragment classification, no taps:", round(float(raw), 3))
    for order in ("margin", "random"):
        c = taps(letters, scores, order)
        n95 = next((n for n, a in c if a >= 0.95), None)
        print(order, "taps->acc:", c[::5] + [c[-1]], "| taps to 95%:", n95,
              "| fragments that could be asked:", len(c) - 1)
    out = {"linking": linking(res, I.tracklets(res, min_n=5), cam)}
    (D / "v2_eval_t2-1.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
