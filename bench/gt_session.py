"""Hand ground truth for robot identity, one fragment at a time, in time order.

usage: gt_session.py KEY [n]      renders the next n unlabelled fragments (default 2)
Labels live in labels/gt_<key>.json: {"fragments": {tid: "A".."F" | "N" | "?"}, ...}.
A..F are the start robots left to right (A-C red, D-F blue). "N" = not a robot, "?" = I
cannot tell even with context. Each strip: 5 frames (t0-1 .. t0+1 s, 0.5 s apart), 560x360
crop around the fragment's first box, every labelled fragment drawn with its letter, the
fragment in question in orange, unlabelled others in grey with their id.
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
L = Path("/home/bodas/data/fgc-vision/labels")
COL = {"A": (60, 60, 255), "B": (0, 160, 255), "C": (200, 0, 255), "D": (255, 160, 0),
       "E": (255, 255, 0), "F": (180, 255, 120), "N": (90, 90, 90), "?": (255, 255, 255)}


def fragments(res):
    return I.tracklets(res, poly=np.array([(300, 250), (1650, 250), (1920, 860), (0, 860)], float),
                       min_n=5)


def main() -> None:
    key = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    res = json.loads((D / "tracks" / f"{key}.pre.json").read_text())
    trs = fragments(res)
    lp = L / f"gt_{key}.json"
    lab = json.loads(lp.read_text()) if lp.exists() else {"fragments": {}}
    done = lab["fragments"]
    todo = [t for t in trs if str(t.id) not in done][:n]
    if not todo:
        print("all", len(trs), "fragments labelled")
        return
    fps, s0 = res["fps"], res["start_frame"]
    cap = cv2.VideoCapture(str(D / "video" / f"2025_{key}.mp4"))
    by_t = {round(fr["t"], 1): fr for fr in res["frames"]}
    alive = {t.id: t for t in trs}
    rows = []
    for t in todo:
        cx, cy = t.c(0)
        x0 = int(np.clip(cx - 280, 0, 1920 - 560))
        y0 = int(np.clip(cy - 200, 0, 1080 - 360))
        tiles = []
        for dt in (-1.0, -0.5, 0.0, 0.5, 1.0):
            tt = round(t.start + dt, 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(s0 + tt * fps)))
            ok, f = cap.read()
            if not ok:
                f = np.zeros((1080, 1920, 3), np.uint8)
            fr = by_t.get(tt) or by_t.get(round(tt + 0.1, 1)) or {"robots": []}
            for tid, a, b, c, d, _ in fr["robots"]:
                if tid not in alive:
                    continue
                lb = done.get(str(tid))
                if tid == t.id:
                    col, txt, w = (0, 140, 255), f"? {tid}", 3
                elif lb:
                    col, txt, w = COL.get(lb, (255, 255, 255)), lb, 2
                else:
                    col, txt, w = (150, 150, 150), str(tid), 1
                cv2.rectangle(f, (int(a), int(b)), (int(c), int(d)), col, w)
                cv2.putText(f, txt, (int(a), int(b) - 4), 0, 0.7, col, 2)
            tile = f[y0:y0 + 360, x0:x0 + 560].copy()
            cv2.putText(tile, f"t={tt:.1f}", (4, 18), 0, 0.55, (0, 255, 255), 2)
            tiles.append(tile)
        row = np.hstack(tiles)
        hdr = np.zeros((26, row.shape[1], 3), np.uint8)
        cv2.putText(hdr, f"fragment {t.id}: {t.start:.1f}-{t.end:.1f}s  start at ({cx:.0f},{cy:.0f})",
                    (4, 19), 0, 0.6, (255, 255, 255), 1)
        rows.append(np.vstack([hdr, row]))
    img = np.vstack(rows)
    scale = min(1.0, 1900 / img.shape[1])
    img = cv2.resize(img, None, fx=scale, fy=scale)
    out = D / "frames" / "gt" / f"{key}_next.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(out, [t.id for t in todo], f"{len(done)}/{len(trs)} labelled")


if __name__ == "__main__":
    main()
