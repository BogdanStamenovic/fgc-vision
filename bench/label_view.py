"""Render a frame with a coordinate grid and a numbered candidate pool for hand labelling.

usage: label_view.py DETS_JSON FRAME_PNG OUT_JPG [cls] [crop x0,y0,x1,y1]
Candidates = union of every model's detections of `cls` (score >= per-model floor),
merged at IoU 0.5. The pool is written next to OUT as <frame>.pool.json so labels can
be given as candidate numbers plus manual boxes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

FLOOR = {"robot": 0.08, "ball": 0.08}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0


def main() -> None:
    dets = json.load(open(sys.argv[1]))
    frame = Path(sys.argv[2])
    out = Path(sys.argv[3])
    cls = sys.argv[4] if len(sys.argv) > 4 else "robot"
    crop = [int(v) for v in sys.argv[5].split(",")] if len(sys.argv) > 5 else [0, 0, 1920, 840]
    pool: list[list[float]] = []
    for m, d in dets.items():
        for b in d["dets"].get(frame.stem, []):
            if b[5] != cls or b[4] < FLOOR[cls] or b[3] > 860 or (b[2] - b[0]) > 500:
                continue
            if not any(iou(b, p) > 0.5 for p in pool):
                pool.append(b[:4])
    im = cv2.imread(str(frame))
    x0, y0, x1, y1 = crop
    for x in range(0, 1920, 50):
        c = (0, 200, 255) if x % 100 == 0 else (0, 120, 160)
        cv2.line(im, (x, 0), (x, 1080), c, 1)
        if x % 100 == 0:
            cv2.putText(im, str(x), (x + 2, y0 + 14), 0, 0.45, (0, 255, 255), 1)
    for y in range(0, 1080, 50):
        c = (0, 200, 255) if y % 100 == 0 else (0, 120, 160)
        cv2.line(im, (0, y), (1920, y), c, 1)
        if y % 100 == 0:
            cv2.putText(im, str(y), (x0 + 2, y - 2), 0, 0.45, (0, 255, 255), 1)
    for i, b in enumerate(pool):
        p = [int(v) for v in b]
        cv2.rectangle(im, (p[0], p[1]), (p[2], p[3]), (255, 0, 255), 2)
        cv2.putText(im, str(i), (p[0], p[1] - 3), 0, 0.6, (255, 255, 255), 2)
    im = im[y0:y1, x0:x1]
    scale = min(1.0, 1500 / im.shape[1])
    im = cv2.resize(im, None, fx=scale, fy=scale)
    cv2.imwrite(str(out), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    out.with_suffix(".pool.json").write_text(json.dumps({"frame": frame.stem, "cls": cls,
                                                         "pool": pool}))
    print(len(pool), "candidates")


if __name__ == "__main__":
    main()
