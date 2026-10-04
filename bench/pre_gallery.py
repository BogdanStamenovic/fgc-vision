"""Start galleries for every pre-start clip in /mnt/offload/fgc-vision/pre/.

For each clip: YOLO (conf >= 0.05) every 1 s, keep robot boxes inside a wide polygon that
covers the rails on all 2025 field cameras, cluster centres (35 px), keep clusters seen in
>= 40 % of frames, take the 3 most persistent per half (left = red). Up to 10 crops per robot.
Embeds with DINOv2-base (CLS + mean patch) and CLIP ViT-L/14; writes pre/galleries.npz.
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
W = str(D / "runs/detect/runs/yolo11s_1280/weights/best.pt")
WIDE = np.array([(380, 270), (1560, 270), (1900, 820), (20, 820)], float)
CROP = 128


def crop(f, b, pad=0.12):
    w, h = b[2] - b[0], b[3] - b[1]
    c = f[int(max(0, b[1] - pad * h)):int(min(f.shape[0], b[3] + pad * h)),
          int(max(0, b[0] - pad * w)):int(min(f.shape[1], b[2] + pad * w))]
    return cv2.resize(c, (CROP, CROP), interpolation=cv2.INTER_CUBIC)


def galleries(model, clip: Path) -> list[dict] | None:
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, dets = {}, []
    for fi in range(0, n, int(fps)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = cap.read()
        if not ok:
            break
        frames[fi] = f
        r = next(iter(model.predict(f, conf=0.05, imgsz=1280, verbose=False)))
        for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if int(k) == 0 and I.inside(WIDE, cx, cy) and b[2] - b[0] < 320 and b[3] < 860:
                dets.append((fi, cx, cy, b, c))
    nf = len(frames)
    if nf < 5:
        return None
    clusters: list[list] = []
    for d in sorted(dets, key=lambda d: -d[4]):
        for cl in clusters:
            if np.hypot(d[1] - cl[0][1], d[2] - cl[0][2]) < 35:
                if all(x[0] != d[0] for x in cl):
                    cl.append(d)
                break
        else:
            clusters.append([d])
    clusters = [c for c in clusters if len(c) >= 0.4 * nf]
    score = lambda cl: len(cl) * np.mean([x[4] for x in cl])  # noqa: E731
    out = []
    for side, grp in (("red", [c for c in clusters if c[0][1] < 960]),
                      ("blue", [c for c in clusters if c[0][1] >= 960])):
        grp = sorted(grp, key=score, reverse=True)[:3]
        for cl in sorted(grp, key=lambda c: c[0][1]):
            sel = sorted(cl, key=lambda x: -x[4])[:10]
            out.append({"side": side, "xy": [round(cl[0][1]), round(cl[0][2])],
                        "seen": round(len(cl) / nf, 2),
                        "crops": np.stack([crop(frames[x[0]], x[3]) for x in sel])})
    return out


def embedders():
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    dm = AutoModel.from_pretrained("facebook/dinov2-base", dtype=torch.float16).eval().cuda()
    dp = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", dtype=torch.float16).eval().cuda()
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def dino(imgs):
        x = dp(images=list(imgs), return_tensors="pt")["pixel_values"].cuda().half()
        with torch.no_grad():
            o = dm(pixel_values=x)
        e = torch.cat([o.last_hidden_state[:, 0], o.last_hidden_state[:, 1:].mean(1)], -1)
        return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()

    def clipl(imgs):
        x = cp(images=list(imgs), return_tensors="pt")["pixel_values"].cuda().half()
        with torch.no_grad():
            e = cm.get_image_features(pixel_values=x)
        e = e if torch.is_tensor(e) else e.pooler_output
        return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
    return {"dinov2-b": dino, "clip-l14": clipl}


def main() -> None:
    from ultralytics import YOLO
    model = YOLO(W)
    emb = embedders()
    rows = []
    for clip in sorted((D / "pre").glob("*.mp4")):
        g = galleries(model, clip)
        key = clip.stem
        if not g:
            print(key, "no frames", flush=True)
            continue
        sides = [x["side"] for x in g]
        print(key, len(g), "robots", sides.count("red"), "red", sides.count("blue"), "blue",
              [x["seen"] for x in g], flush=True)
        for j, x in enumerate(g):
            rgb = x["crops"][..., ::-1].copy()
            rows.append({"key": key, "j": j, "side": x["side"], "xy": x["xy"], "seen": x["seen"],
                         **{k: f(rgb) for k, f in emb.items()}, "crops": x["crops"][:4]})
    np.savez_compressed(D / "pre" / "galleries.npz",
                        meta=json.dumps([{k: r[k] for k in ("key", "j", "side", "xy", "seen")}
                                         for r in rows]),
                        crops=np.stack([r["crops"] if len(r["crops"]) == 4 else
                                        np.concatenate([r["crops"]] * 4)[:4] for r in rows]),
                        **{k: np.stack([r[k].mean(0) for r in rows]) for k in emb})
    print("rows", len(rows))


if __name__ == "__main__":
    main()
