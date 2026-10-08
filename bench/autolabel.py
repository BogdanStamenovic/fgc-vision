"""Pseudo-label frames with OWLv2 and write a YOLO dataset.

usage: autolabel.py OUT_DIR clip.mp4[,clip2.mp4...] [every_s]
Robots kept at OWLv2 score >= 0.25 (the F1-optimal threshold on the hand labels was 0.225),
balls at >= 0.30. Boxes wider than 320 px, or below y=860 (overlay), are dropped.
Labels are NOT hand-corrected: the fine-tuned model learns OWLv2's errors too, and the
evaluation on hand-labelled frames from other clips is what says whether that matters.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from detect_bench import owlv2  # noqa: E402

THR = {"robot": float(__import__("os").environ.get("ROBOT_THR", 0.25)), "ball": 0.30}
CLS = {"robot": 0, "ball": 1}


def main() -> None:
    out = Path(sys.argv[1])
    clips = sys.argv[2].split(",")
    every = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    run = owlv2()
    random.seed(0)
    n = 0
    for c in clips:
        cap = cv2.VideoCapture(c)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = int(fps * every)
        for fi in range(int(fps * 10), total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, f = cap.read()
            if not ok:
                break
            split = "val" if random.random() < 0.1 else "train"
            stem = f"{Path(c).stem}_{fi}"
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), f)
            img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            h, w = f.shape[:2]
            rows = []
            for x0, y0, x1, y1, s, cls in run(img):
                if s < THR[cls] or x1 - x0 > 320 or y1 > 860:
                    continue
                rows.append(f"{CLS[cls]} {(x0+x1)/2/w:.6f} {(y0+y1)/2/h:.6f} "
                            f"{(x1-x0)/w:.6f} {(y1-y0)/h:.6f}")
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(rows))
            n += 1
        print(c, n, file=sys.stderr, flush=True)
    (out / "data.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\n"
                                   "names: {0: robot, 1: ball}\n")


if __name__ == "__main__":
    main()
