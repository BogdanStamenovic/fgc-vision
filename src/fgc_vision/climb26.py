"""2026 BRACE climb detection from tracker output.

The two BRACE pipes run from the guardrail up to the top corners of the EXTINGUISHER.
Each pipe is a hand-clicked image segment per camera (bottom -> top). A robot sample is
"on the pipe" when its box centre is within `band` px of the segment; its progress s is the
projection onto the segment (0 = guardrail end, 1 = top). A climb is a run of on-pipe
samples (gaps up to `gap` s allowed, any track id: the tracker loses climbers in the
crowd at the top) whose s rises by at least `min_rise`; it starts at the last sample with
s within 0.05 of the run's minimum before the rise, and ends when s first reaches 95 % of
the run's maximum.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def on_pipe(cx: float, cy: float, seg: list[float], band: float = 70.0) -> float | None:
    x0, y0, x1, y1 = seg
    d = np.array([x1 - x0, y1 - y0], float)
    L2 = float(d @ d)
    s = float(((cx - x0) * d[0] + (cy - y0) * d[1]) / L2)
    if s < -0.05 or s > 1.1:
        return None
    px, py = x0 + s * d[0], y0 + s * d[1]
    return s if np.hypot(cx - px, cy - py) <= band else None


def climbs(res: dict[str, Any], seg: list[float], band: float = 70.0, gap: float = 2.0,
           min_rise: float = 0.3, t_from: float = 60.0) -> list[dict[str, float]]:
    pts = []
    for fr in res["frames"]:
        if fr["t"] < t_from:
            continue
        for r in fr["robots"]:
            s = on_pipe((r[1] + r[3]) / 2, (r[2] + r[4]) / 2, seg, band)
            if s is not None:
                pts.append((fr["t"], s, r[0]))
    pts.sort()
    # group into per-climber runs: greedy, a sample continues the run whose last sample is
    # closest in s within `gap` seconds (several robots can be on one pipe)
    runs: list[list[tuple[float, float, int]]] = []
    for p in pts:
        best, bd = None, 0.12
        for r in runs:
            if p[0] - r[-1][0] <= gap and abs(p[1] - r[-1][1]) < bd:
                best, bd = r, abs(p[1] - r[-1][1])
        if best is None:
            runs.append([p])
        else:
            best.append(p)
    out = []
    for r in runs:
        a = np.array([(t, s) for t, s, _ in r])
        if len(a) < 8:
            continue
        k = 5
        sm = np.convolve(a[:, 1], np.ones(k) / k, "valid")
        t = a[k // 2:k // 2 + len(sm), 0]
        if sm.max() - sm[: max(1, int(np.argmax(sm)))].min() < min_rise:
            continue
        imax = int(np.argmax(sm))
        imin = int(np.argmin(sm[: imax + 1]))
        lo = sm[imin]
        i0 = max(i for i in range(imin, imax + 1) if sm[i] <= lo + 0.05)
        i1 = next(i for i in range(i0, imax + 1) if sm[i] >= lo + 0.95 * (sm[imax] - lo))
        out.append({"start": round(float(t[i0]), 1), "end": round(float(t[i1]), 1),
                    "seconds": round(float(t[i1] - t[i0]), 1),
                    "s_from": round(float(lo), 2), "s_to": round(float(sm[imax]), 2),
                    "samples": len(a)})
    return sorted(out, key=lambda c: c["end"])
