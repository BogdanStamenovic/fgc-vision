"""Read the broadcast overlay: match timer and live score.

The FGC field streams carry the same scorebug on every field (2025): timer centred at
x 850-1070, y 855-950 of a 1920x1080 frame, red/blue running score below it, station
order top-to-bottom = stations x1, x2, x3 (checked on t2-1: GRN/BHU/ALB = 11/12/13).
Tesseract with a digit whitelist reads it; Otsu binarisation handles the orange/red/blue
backgrounds. Positions are for 1080p and scale linearly.

The timer counts down from 2:30 and shows ceil(remaining), so the 2:30 -> 2:29 flip
happens one second after the start. That flip pins match time to about one frame step.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

ROI = {"timer": (850, 855, 1070, 950), "red": (855, 960, 955, 1055),
       "blue": (965, 960, 1065, 1055)}
MATCH_S = 150
# 2026 scorebug: same place, but a black frame sits inside the 2025 box and breaks Otsu.
ROI_TIMER = {2025: (850, 855, 1070, 950), 2026: (868, 868, 1035, 945)}


def use_year(year: int) -> None:
    ROI["timer"] = ROI_TIMER.get(year, ROI_TIMER[2026])


class OverlayError(RuntimeError):
    pass


def _ocr(img: np.ndarray, whitelist: str) -> str:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if b.mean() < 128:
        b = 255 - b
    b = cv2.copyMakeBorder(b, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    _ok, png = cv2.imencode(".png", b)
    p = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "7", "-c",
                        f"tessedit_char_whitelist={whitelist}"],
                       input=png.tobytes(), capture_output=True, check=False)
    return p.stdout.decode().strip()


def _crop(frame: np.ndarray, key: str) -> np.ndarray:
    h, w = frame.shape[:2]
    sx, sy = w / 1920, h / 1080
    x0, y0, x1, y1 = ROI[key]
    return frame[int(y0 * sy):int(y1 * sy), int(x0 * sx):int(x1 * sx)]


def remaining(frame: np.ndarray) -> int | None:
    """Seconds left on the timer, or None if it does not read as m:ss."""
    t = _ocr(_crop(frame, "timer"), "0123456789:")
    m = re.fullmatch(r"([0-2]):?([0-5]\d)", t)
    if not m:
        return None
    s = int(m.group(1)) * 60 + int(m.group(2))
    return s if s <= MATCH_S else None


def scores(frame: np.ndarray) -> tuple[int | None, int | None]:
    out = []
    for k in ("red", "blue"):
        t = _ocr(_crop(frame, k), "0123456789")
        out.append(int(t) if t.isdigit() and len(t) <= 3 else None)
    return out[0], out[1]


@dataclass
class Sync:
    start_frame: int      # frame index of match t=0 in the clip
    fps: float
    flips_seen: int       # timer readings consistent with the fit
    readings: int

    def t(self, frame_index: int) -> float:
        return (frame_index - self.start_frame) / self.fps


def sync(path: str, log: Callable[[str], None] = lambda s: None) -> Sync:
    """Fit match start from timer readings at 1 Hz, refined at the 2:30->2:29 flip."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pts: list[tuple[int, int]] = []
    for fi in range(0, n, round(fps)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = cap.read()
        if not ok:
            break
        r = remaining(f)
        if r is not None and 0 < r < MATCH_S:
            pts.append((fi, r))
    if len(pts) < 10:
        raise OverlayError(f"timer readable in only {len(pts)} frames of {path}")
    # start = fi - (150 - r) * fps, up to one second of ceil() slack: take the median
    # estimate, then keep readings that agree within 1.5 s.
    est = np.array([fi - (MATCH_S - r) * fps for fi, r in pts])
    med = float(np.median(est))
    good = [(fi, r) for (fi, r), e in zip(pts, est) if abs(e - med) < 1.5 * fps]
    # refine at the first flip below 2:30: the frame where 2:29 first appears is t=1 s
    first = min((fi for fi, r in good if r == MATCH_S - 1), default=None)
    start = med - fps  # ceil() means the estimate above sits one second late on average
    if first is not None:
        step = max(1, int(fps / 10))
        lo = max(0, first - int(fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        fi = lo
        while fi <= first:
            ok, f = cap.read()
            if not ok:
                break
            if (fi - lo) % step == 0 and remaining(f) == MATCH_S - 1:
                start = fi - fps
                break
            fi += 1
    log(f"sync {path}: start frame {start:.0f} ({start / fps:.2f}s), "
        f"{len(good)}/{len(pts)} readings consistent")
    return Sync(round(start), fps, len(good), len(pts))
