"""Extract per-robot start galleries and per-fragment crops for one match.

usage: gallery_extract.py KEY
Gallery: YOLO detections (conf >= 0.05) on frames every 0.5 s in the pre-start window
[-12, -0.5] s, clustered by centre (robots are stationary there). The 3 most persistent
clusters in each half of the field are the 6 robots; left half = red, right = blue.
Fragments: every tracklet from tracks/<key>.pre.json (field mask, static filter, >= 10
samples) that has samples after t = -0.5 s; 8 crops evenly spread over its life.
Writes /mnt/offload/fgc-vision/reid/<key>.npz and a labelling montage (no predictions on it).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/bodas/data/fgc-vision/src")
from fgc_vision import identity as I  # noqa: E402

D = Path("/mnt/offload/fgc-vision")
W = str(D / "runs/detect/runs/yolo11s_1280/weights/best.pt")
CROP = 128


def crop(f, b, pad=0.12):
    w, h = b[2] - b[0], b[3] - b[1]
    c = f[int(max(0, b[1] - pad * h)):int(min(f.shape[0], b[3] + pad * h)),
          int(max(0, b[0] - pad * w)):int(min(f.shape[1], b[2] + pad * w))]
    return cv2.resize(c, (CROP, CROP), interpolation=cv2.INTER_CUBIC)


def main() -> None:
    from ultralytics import YOLO
    key = sys.argv[1]
    res = json.loads((D / "tracks" / f"{key}.pre.json").read_text())
    fps, s0 = res["fps"], res["start_frame"]
    cap = cv2.VideoCapture(str(D / "video" / f"2025_{key}.mp4"))

    def frame(t):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(s0 + t * fps)))
        return cap.read()[1]

    model = YOLO(W)
    t_pre0 = max(-12.0, -s0 / fps + 0.2)
    dets = []
    for t in np.arange(t_pre0, -0.4, 0.5):
        f = frame(t)
        r = next(iter(model.predict(f, conf=0.05, imgsz=1280, verbose=False)))
        for (x0, y0, x1, y1), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                          r.boxes.cls.tolist()):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if int(k) == 0 and I.inside(I.FIELD_POLY_2025, cx, cy) and x1 - x0 < 320:
                dets.append((float(t), cx, cy, [x0, y0, x1, y1], c))
    nt = len(np.arange(t_pre0, -0.4, 0.5))
    # greedy clustering by centre, 35 px
    clusters: list[list] = []
    for d in sorted(dets, key=lambda d: -d[4]):
        for cl in clusters:
            if np.hypot(d[1] - cl[0][1], d[2] - cl[0][2]) < 35:
                if all(abs(x[0] - d[0]) > 1e-6 for x in cl):
                    cl.append(d)
                break
        else:
            clusters.append([d])
    score = lambda cl: len(cl) / nt * np.mean([x[4] for x in cl])  # noqa: E731
    left = sorted([c for c in clusters if c[0][1] < 960], key=score, reverse=True)[:3]
    right = sorted([c for c in clusters if c[0][1] >= 960], key=score, reverse=True)[:3]
    gal = []
    for side, group in (("red", left), ("blue", right)):
        for cl in sorted(group, key=lambda c: c[0][1]):
            crops = [crop(frame(t), b) for t, _, _, b, _ in sorted(cl)[::max(1, len(cl) // 12)]]
            gal.append({"side": side, "xy": [round(cl[0][1]), round(cl[0][2])],
                        "seen": round(len(cl) / nt, 2), "crops": np.stack(crops)})
    trs = [t for t in I.tracklets(res, min_n=10) if t.end > -0.5]
    frags = []
    for t in trs:
        idx = np.linspace(0, len(t.t) - 1, min(8, len(t.t))).astype(int)
        crops = np.stack([crop(frame(t.t[i]), t.box[i]) for i in idx])
        frags.append({"id": t.id, "t0": t.start, "t1": t.end, "n": len(t.t),
                      "c0": t.c(0), "c1": t.c(-1), "crops": crops})
    out = D / "reid"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / f"{key}.npz",
                        gal=np.stack([g["crops"][:12] if len(g["crops"]) >= 12 else
                                      np.concatenate([g["crops"]] * 12)[:12] for g in gal]),
                        frag=np.stack([f["crops"] if len(f["crops"]) == 8 else
                                       np.concatenate([f["crops"]] * 8)[:8] for f in frags]),
                        meta=json.dumps({"gal": [{k: v for k, v in g.items() if k != "crops"}
                                                 for g in gal],
                                         "frag": [{k: v for k, v in f.items() if k != "crops"}
                                                  for f in frags]}))
    print(f"{key}: gallery {[(g['side'], g['xy'], g['seen']) for g in gal]}; "
          f"{len(frags)} fragments")
    # labelling montage: gallery row (G0..G5, 3 crops each) then fragments (3 crops each)
    S = 96
    rows = []
    hdr = np.full((S + 18, 6 * (3 * S + 8), 3), 30, np.uint8)
    for i, g in enumerate(gal):
        for j in range(3):
            c = cv2.resize(g["crops"][j * (len(g["crops"]) - 1) // 2], (S, S))
            hdr[18:, i * (3 * S + 8) + j * S: i * (3 * S + 8) + (j + 1) * S] = c
        cv2.putText(hdr, f"G{i} {g['side']} {g['xy']}", (i * (3 * S + 8), 13), 0, 0.45,
                    (0, 255, 255), 1)
    cols = 6
    for k in range(0, len(frags), 36):
        chunk = frags[k:k + 36]
        nr = (len(chunk) + cols - 1) // cols
        body = np.full((nr * (S + 18), cols * (3 * S + 8), 3), 50, np.uint8)
        for i, fr in enumerate(chunk):
            r, c = divmod(i, cols)
            for j, ci in enumerate((0, 3, 7)):
                body[r * (S + 18) + 18: (r + 1) * (S + 18),
                     c * (3 * S + 8) + j * S: c * (3 * S + 8) + (j + 1) * S] = \
                    cv2.resize(fr["crops"][ci], (S, S))
            cv2.putText(body, f"F{k + i} {fr['t0']:.0f}-{fr['t1']:.0f}s", (c * (3 * S + 8),
                        r * (S + 18) + 13), 0, 0.45, (255, 255, 255), 1)
        cv2.imwrite(str(out / f"{key}.label_{k // 36}.jpg"), np.vstack([hdr, body]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
