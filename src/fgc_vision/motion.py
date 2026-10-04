"""Driving metrics from a field-coordinate track (t, x, y in metres).

Smoothed with a centred moving average over `win` samples (10 Hz input: 0.5 s) before
differencing, because box bottom-centres jitter by ~5-10 cm frame to frame and raw
differences turn that into fake speed. Gaps longer than `max_gap` s are not bridged:
path and speed are summed only within continuous runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def metrics(trk: np.ndarray, win: int = 5, max_gap: float = 0.35, idle_speed: float = 0.1,
            field: tuple[float, float] = (7.0, 7.0)) -> dict[str, Any]:
    trk = trk[np.argsort(trk[:, 0], kind="stable")]
    # overlapping fragments of one robot give duplicate timestamps; keep the first
    trk = trk[np.r_[True, np.diff(trk[:, 0]) > 1e-6]]
    if len(trk) < win + 2:
        return {"samples": len(trk), "seconds_observed": 0.0}
    runs = np.split(trk, np.nonzero(np.diff(trk[:, 0]) > max_gap)[0] + 1)
    path = obs = idle = 0.0
    speeds, accs, jerks = [], [], []
    for r in runs:
        if len(r) < win + 2:
            continue
        k = np.ones(win) / win
        x = np.convolve(r[:, 1], k, "valid")
        y = np.convolve(r[:, 2], k, "valid")
        t = r[win // 2: win // 2 + len(x), 0]
        dt = np.diff(t)
        v = np.hypot(np.diff(x), np.diff(y)) / dt
        path += float(np.hypot(np.diff(x), np.diff(y)).sum())
        obs += float(t[-1] - t[0])
        idle += float(dt[v < idle_speed].sum())
        speeds.append(v)
        if len(v) > 2:
            a = np.diff(v) / dt[1:]
            accs.append(a)
            if len(a) > 2:
                jerks.append(np.diff(a) / dt[2:])
    if not speeds:
        return {"samples": len(trk), "seconds_observed": 0.0}
    v = np.concatenate(speeds)
    a = np.concatenate(accs) if accs else np.zeros(1)
    j = np.concatenate(jerks) if jerks else np.zeros(1)
    inside = ((trk[:, 1] > -0.3) & (trk[:, 1] < field[0] + 0.3) &
              (trk[:, 2] > -0.3) & (trk[:, 2] < field[1] + 0.3)).mean()
    return {"samples": len(trk), "seconds_observed": round(obs, 1),
            "path_m": round(path, 2), "mean_speed_mps": round(path / obs, 2) if obs else None,
            "p90_speed_mps": round(float(np.percentile(v, 90)), 2),
            "p90_abs_accel_mps2": round(float(np.percentile(np.abs(a), 90)), 2),
            "rms_jerk_mps3": round(float(np.sqrt((j ** 2).mean())), 1),
            "idle_seconds": round(idle, 1), "share_inside_field": round(float(inside), 3)}
