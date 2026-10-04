"""Time-sampled identity ground truth (replaces per-fragment labels, because a tracker
fragment can switch robots in the middle: t2-1 fragment 2349 starts on robot C and ends on B).

usage: gt_points.py KEY render T1 [T2]   -> frames/gt/<key>_pts.jpg
       gt_points.py KEY set T i=L ...    -> labels/gtp_<key>.json
Each sheet row: the 6 start robots (A-F, 200 px upscaled), then every tracked robot box in the
field at time T, numbered, upscaled to 200 px, and a small full-field context frame with the
numbers. Labels: A..F, N (not a robot), ? (cannot tell), D2 (duplicate box of a robot already
listed at this time; ignored in scoring).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/bodas/data/fgc-vision/src")
sys.path.insert(0, "/home/bodas/data/fgc-vision/bench")
from gt_session import fragments  # noqa: E402

D = Path("/mnt/offload/fgc-vision")
L = Path("/home/bodas/data/fgc-vision/labels")
S = 200
START = {"t2-1": {"A": 3, "B": 1, "C": 6, "D": 7, "E": 90, "F": 10}}


def boxes_at(res, trs, t):
    ids = {x.id for x in trs}
    fr = min(res["frames"], key=lambda f: abs(f["t"] - t))
    return fr["t"], [r for r in fr["robots"] if r[0] in ids]


def crop(f, b, s=S):
    x0, y0, x1, y1 = b[:4]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h = max(x1 - x0, y1 - y0) * 0.75 + 6
    c = f[int(max(0, cy - h)):int(min(1080, cy + h)), int(max(0, cx - h)):int(min(1920, cx + h))]
    return cv2.resize(c, (s, s), interpolation=cv2.INTER_CUBIC)


def render(key, ts):
    res = json.loads((D / "tracks" / f"{key}.pre.json").read_text())
    trs = fragments(res)
    cap = cv2.VideoCapture(str(D / "video" / f"2025_{key}.mp4"))
    s0, fps = res["start_frame"], res["fps"]

    def frame(t):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(s0 + t * fps)))
        return cap.read()[1]
    f0 = frame(-5.0)
    _, b0 = boxes_at(res, trs, -5.0)
    bmap = {r[0]: r[1:5] for r in b0}
    hdr = []
    for k, tid in START[key].items():
        c = crop(f0, bmap[tid]) if tid in bmap else np.zeros((S, S, 3), np.uint8)
        cv2.putText(c, k, (6, 30), 0, 1.1, (0, 255, 255), 3)
        hdr.append(c)
    rows = [np.hstack(hdr)]
    for t in ts:
        tt, bs = boxes_at(res, trs, t)
        f = frame(tt)
        ctx = f.copy()
        tiles = []
        for i, r in enumerate(bs):
            c = crop(f, r[1:5])
            cv2.putText(c, str(i), (6, 30), 0, 1.1, (0, 140, 255), 3)
            tiles.append(c)
            cv2.rectangle(ctx, (int(r[1]), int(r[2])), (int(r[3]), int(r[4])), (0, 140, 255), 3)
            cv2.putText(ctx, str(i), (int(r[1]), int(r[2]) - 6), 0, 1.4, (0, 140, 255), 4)
        ctx = cv2.resize(ctx[230:880], (int(1920 * S * 2 / 650 / 2), S))
        cv2.putText(ctx, f"t={tt:.1f}", (6, 24), 0, 0.8, (0, 255, 255), 2)
        row = np.hstack([ctx] + tiles)
        rows.append(row)
    w = max(r.shape[1] for r in rows)
    img = np.vstack([np.pad(r, ((0, 4), (0, w - r.shape[1]), (0, 0))) for r in rows])
    scale = min(1.0, 2000 / img.shape[1])
    img = cv2.resize(img, None, fx=scale, fy=scale)
    out = D / "frames" / "gt" / f"{key}_pts.jpg"
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    meta = {str(t): [r[0] for r in boxes_at(res, trs, t)[1]] for t in ts}
    print(out, json.dumps(meta))


def setl(key, t, kvs):
    p = L / f"gtp_{key}.json"
    d = json.loads(p.read_text()) if p.exists() else {
        "_doc": __doc__.split("\n\n")[0], "start": START[key], "points": {}}
    res = json.loads((D / "tracks" / f"{key}.pre.json").read_text())
    trs = fragments(res)
    tt, bs = boxes_at(res, trs, float(t))
    pts = d["points"].setdefault(f"{tt:.1f}", {})
    for kv in kvs:
        i, lab = kv.split("=")
        pts[str(bs[int(i)][0])] = lab
    p.write_text(json.dumps(d, indent=1))
    print(f"t={tt:.1f}:", pts, f"({sum(len(v) for v in d['points'].values())} labels)")


if __name__ == "__main__":
    key, cmd = sys.argv[1], sys.argv[2]
    if cmd == "render":
        render(key, [float(x) for x in sys.argv[3:]])
    else:
        setl(key, sys.argv[3], sys.argv[4:])
