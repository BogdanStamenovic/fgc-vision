"""Time-sampled identity ground truth on 2026 clips.

usage: gt26.py KEY start                -> frames/gt/<key>_start.jpg (numbered pre-start boxes)
       gt26.py KEY seeds i=A j=B ...    -> labels/gt26_<key>.json start robots (A-C red, D-F blue)
       gt26.py KEY render T1 T2 ...     -> frames/gt/<key>_pts.jpg
       gt26.py KEY set T i=L ...        -> labels (A..F, N not a robot, ? can't tell, D2 duplicate)
       gt26.py KEY hidden T A B ...     -> robots I can see in the frame at T but that have no box
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

D = Path("/mnt/offload/fgc-vision/2026")
L = Path("/home/bodas/data/fgc-vision/labels")
S = 180
# 2026 field cameras: the carpet plus the pipes up to the goals; excludes the crowd and
# the scorebug. One polygon fits fields 2 and 3 (checked on t2-46/63/66).
POLY = np.array([(300, 180), (1650, 180), (1820, 760), (100, 760)], np.int32)


def load(key):
    res = json.loads((D / "tracks" / f"{key}.json").read_text())
    p = L / f"gt26_{key}.json"
    lab = json.loads(p.read_text()) if p.exists() else {"start": {}, "points": {}, "hidden": {}}
    return res, lab, p


def boxes_at(res, t):
    fr = min(res["frames"], key=lambda f: abs(f["t"] - t))
    bs = [r for r in fr["robots"]
          if cv2.pointPolygonTest(POLY, ((r[1] + r[3]) / 2, (r[2] + r[4]) / 2), False) >= 0]
    return fr["t"], sorted(bs, key=lambda r: r[1])


def frame(key, res, t):
    cap = cv2.VideoCapture(str(D / f"{key}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(res["start_frame"] + t * res["fps"])))
    return cap.read()[1]


def crop(f, b, s=S):
    x0, y0, x1, y1 = b[:4]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h = max(x1 - x0, y1 - y0) * 0.75 + 8
    c = f[int(max(0, cy - h)):int(min(1080, cy + h)), int(max(0, cx - h)):int(min(1920, cx + h))]
    return cv2.resize(c, (s, s), interpolation=cv2.INTER_CUBIC)


def ctx(f, bs, t):
    g = f.copy()
    for i, r in enumerate(bs):
        cv2.rectangle(g, (int(r[1]), int(r[2])), (int(r[3]), int(r[4])), (0, 140, 255), 3)
        cv2.putText(g, str(i), (int(r[1]), int(r[2]) - 6), 0, 1.4, (0, 140, 255), 4)
    cv2.putText(g, f"t={t:.1f}", (320, 230), 0, 1.4, (0, 255, 255), 3)
    return cv2.resize(g[170:880, 100:1820], (860, 355))


def main() -> None:
    key, cmd = sys.argv[1], sys.argv[2]
    res, lab, p = load(key)
    if cmd == "start":
        t, bs = boxes_at(res, -3.0)
        f = frame(key, res, t)
        strip = np.hstack([crop(f, r[1:5], 143) for r in bs] +
                          [np.zeros((143, 143, 3), np.uint8)] * max(0, 6 - len(bs)))
        strip = cv2.resize(strip, (860, int(143 * 860 / strip.shape[1])))
        img = np.vstack([ctx(f, bs, t), strip])
        cv2.imwrite(str(D.parent / "frames/gt" / f"{key}_start.jpg"), img)
        print([(i, r[0], round((r[1] + r[3]) / 2), round(r[4])) for i, r in enumerate(bs)])
    elif cmd == "seeds":
        t, bs = boxes_at(res, -3.0)
        for kv in sys.argv[3:]:
            i, L_ = kv.split("=")
            lab["start"][L_] = bs[int(i)][1:5]
        p.write_text(json.dumps(lab, indent=1))
        print(lab["start"])
    elif cmd == "render":
        f0 = frame(key, res, -3.0)
        hdr = []
        for k in "ABCDEF":
            c = crop(f0, lab["start"][k]) if k in lab["start"] else np.zeros((S, S, 3), np.uint8)
            cv2.putText(c, k, (6, 30), 0, 1.1, (0, 255, 255), 3)
            hdr.append(c)
        rows = [cv2.resize(np.hstack(hdr), (860, int(S * 860 / (6 * S))))]
        meta = {}
        for t in map(float, sys.argv[3:]):
            tt, bs = boxes_at(res, t)
            f = frame(key, res, tt)
            tiles = []
            for i, r in enumerate(bs):
                c = crop(f, r[1:5])
                cv2.putText(c, str(i), (6, 30), 0, 1.1, (0, 140, 255), 3)
                tiles.append(c)
            if tiles:
                st = np.hstack(tiles + [np.zeros((S, S, 3), np.uint8)] * max(0, 5 - len(tiles)))
                st = cv2.resize(st, (860, int(S * 860 / st.shape[1])))
                rows.append(np.vstack([ctx(f, bs, tt), st]))
            else:
                rows.append(ctx(f, bs, tt))
            meta[f"{tt:.1f}"] = [r[0] for r in bs]
        w = max(r.shape[1] for r in rows)
        img = np.vstack([np.pad(r, ((0, 4), (0, w - r.shape[1]), (0, 0))) for r in rows])
        cv2.imwrite(str(D.parent / "frames/gt" / f"{key}_pts.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(meta)
    elif cmd == "set":
        tt, bs = boxes_at(res, float(sys.argv[3]))
        pts = lab["points"].setdefault(f"{tt:.1f}", {})
        for kv in sys.argv[4:]:
            i, L_ = kv.split("=")
            pts[str(bs[int(i)][0])] = L_
        p.write_text(json.dumps(lab, indent=1))
        print(f"t={tt:.1f}", pts)
    elif cmd == "hidden":
        tt, _ = boxes_at(res, float(sys.argv[3]))
        lab["hidden"][f"{tt:.1f}"] = sys.argv[4:]
        p.write_text(json.dumps(lab, indent=1))


if __name__ == "__main__":
    main()
