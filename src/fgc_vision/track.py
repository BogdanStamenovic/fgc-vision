"""Detect and track robots (and count ball detections) through one match clip.

The detector is the fine-tuned YOLO (classes robot=0, ball=1). Tracking runs on every
`step`-th frame (default 6 -> 10 Hz on 60 fps video) with Ultralytics' ByteTrack or
BoT-SORT. The track buffer is raised from its default 30 frames (3 s at 10 Hz) to
`buffer`, because robots routinely vanish behind the towers or a referee for 3-8 s.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2

from .overlay import MATCH_S, Sync

TRACKER_TMPL = {
    "bytetrack": ("tracker_type: bytetrack\ntrack_high_thresh: {hi}\ntrack_low_thresh: 0.1\n"
                  "new_track_thresh: {hi}\ntrack_buffer: {buf}\nmatch_thresh: 0.8\n"
                  "fuse_score: True\n"),
    "botsort": ("tracker_type: botsort\ntrack_high_thresh: {hi}\ntrack_low_thresh: 0.1\n"
                "new_track_thresh: {hi}\ntrack_buffer: {buf}\nmatch_thresh: 0.8\n"
                "fuse_score: True\ngmc_method: none\nproximity_thresh: 0.5\n"
                "appearance_thresh: 0.25\nwith_reid: False\nmodel: auto\n"),
}


def tracker_yaml(kind: str, buf: int, hi: float, where: Path) -> str:
    p = where / f"{kind}_{buf}_{hi}.yaml"
    p.write_text(TRACKER_TMPL[kind].format(buf=buf, hi=hi))
    return str(p)


def run(clip: str, weights: str, sync: Sync, kind: str = "bytetrack", step: int = 6,
        buffer: int = 100, conf: float = 0.25, device: str | int = 0, imgsz: int = 1280,
        pre_s: float = 2.0, post_s: float = 6.0,
        log: Callable[[str], None] = lambda s: None) -> dict[str, Any]:
    from ultralytics import YOLO
    model = YOLO(weights)
    cfg = tracker_yaml(kind, buffer, conf, Path(clip).parent)
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS)
    first = max(0, int(sync.start_frame - pre_s * fps))
    last = int(sync.start_frame + (MATCH_S + post_s) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames: list[dict[str, Any]] = []
    fi = first
    t_det = 0.0
    t0 = time.perf_counter()
    while fi <= last:
        ok, f = cap.read()
        if not ok:
            break
        if (fi - first) % step == 0:
            t1 = time.perf_counter()
            r = next(iter(model.track(f, persist=True, tracker=cfg, conf=0.1, imgsz=imgsz,
                                      device=device, verbose=False)))
            t_det += time.perf_counter() - t1
            robots, balls = [], []
            b = r.boxes
            assert b is not None
            ids = b.id.tolist() if b.id is not None else [None] * len(b)
            for xyxy, c, k, i in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist(), ids):
                if xyxy[3] > 860:      # scorebug
                    continue
                if int(k) == 0 and i is not None:
                    robots.append([int(i), *[round(v, 1) for v in xyxy], round(c, 3)])
                elif int(k) == 1 and c >= 0.3:
                    balls.append([round(v, 1) for v in xyxy] + [round(c, 3)])
            frames.append({"f": fi, "t": round(sync.t(fi), 3), "robots": robots, "balls": balls})
        fi += 1
    wall = time.perf_counter() - t0
    n = len(frames)
    log(f"{clip}: {n} frames tracked, {t_det / max(n, 1) * 1000:.1f} ms/frame model+tracker, "
        f"{wall / max(n, 1) * 1000:.1f} ms/frame incl. decode")
    return {"clip": clip, "weights": weights, "tracker": kind, "buffer": buffer, "step": step,
            "fps": fps, "start_frame": sync.start_frame, "frames": frames,
            "ms_per_frame_model": t_det / max(n, 1) * 1000,
            "ms_per_frame_total": wall / max(n, 1) * 1000}


def save(res: dict[str, Any], out: Path) -> None:
    out.write_text(json.dumps(res))
