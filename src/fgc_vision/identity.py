"""Turn fragmented tracker output into one chain per robot, and anchor chains to stations.

Why a stitcher: on 2025 field streams every robot disappears behind the three towers
or a referee every few seconds, so ByteTrack/BoT-SORT produce ~10 tracklets per robot
(t2-1: 58 tracklets >= 2 s for 6 robots). Raising the track buffer does not fix it,
because the robot reappears tens of pixels away on the other side of a tower.

Model: there are exactly N robots (6, or 8 in playoffs) for the whole match. Chains are
seeded from the tracklets present at the start (start anchor: which half of the field a
robot starts in gives its alliance). Every later tracklet is appended to the free chain
whose last position is closest, if the jump is physically plausible (gated by gap time).
Tracklets that fit no chain are dropped (they are usually false positives).

End anchor: a robot that climbed shows a sustained upward image motion with little
sideways motion in the last ~30 s. Official per-robot end-game values then say which
station climbed; when the climbers in an alliance are unique the chain gets the station.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Field-floor polygon (plus room above the centre for hanging robots) on the 2025 side-
# field cameras (fields 1 and 3 checked). Outside it: the arena wall, tower tops, people.
FIELD_POLY_2025 = np.array([(560, 290), (1380, 290), (1830, 770), (110, 770)], float)


def inside(poly: np.ndarray, x: float, y: float) -> bool:
    n = len(poly)
    c = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi:
            c = not c
        j = i
    return c


@dataclass
class Tracklet:
    id: int
    t: list[float] = field(default_factory=list)
    box: list[list[float]] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.t[0]

    @property
    def end(self) -> float:
        return self.t[-1]

    def c(self, i: int) -> tuple[float, float]:
        b = self.box[i]
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def tracklets(res: dict[str, Any], poly: np.ndarray = FIELD_POLY_2025,
              min_n: int = 5) -> list[Tracklet]:
    T: dict[int, Tracklet] = {}
    for fr in res["frames"]:
        for tid, x0, y0, x1, y1, _c in fr["robots"]:
            if not inside(poly, (x0 + x1) / 2, (y0 + y1) / 2):
                continue
            tr = T.setdefault(tid, Tracklet(tid))
            tr.t.append(fr["t"])
            tr.box.append([x0, y0, x1, y1])
    return sorted((t for t in T.values() if len(t.t) >= min_n and not static(t)),
                  key=lambda t: t.start)


def static(t: Tracklet, min_s: float = 10.0, max_px: float = 10.0) -> bool:
    """A box that sits still for 10 s or more is field furniture (AprilTags on the
    towers, a wall sign), not a robot. Cost: a robot parked still that long is
    dropped until it moves again; it re-enters as a new tracklet."""
    if t.end - t.start < min_s:
        return False
    c = np.array([t.c(i) for i in range(len(t.t))])
    return bool(np.ptp(c[:, 0]) < max_px and np.ptp(c[:, 1]) < max_px)


@dataclass
class Chain:
    slot: int
    side: str                       # 'red' (left half at start) or 'blue'
    parts: list[Tracklet] = field(default_factory=list)

    @property
    def end(self) -> float:
        return self.parts[-1].end

    def last(self) -> tuple[float, float]:
        return self.parts[-1].c(-1)

    def samples(self) -> list[tuple[float, list[float]]]:
        return [(t, b) for p in self.parts for t, b in zip(p.t, p.box)]


def stitch(trs: list[Tracklet], n_robots: int = 6, mid_x: float = 960.0,
           seed_window: float = 3.0, max_gap: float = 15.0, base_px: float = 90.0,
           px_per_s: float = 220.0, overlap: float = 0.3) -> list[Chain]:
    # seeds: the n_robots longest-lived tracklets that exist in the first seconds and do
    # not overlap in space (two tracklets on one robot at the start are a duplicate)
    early = [t for t in trs if t.start <= seed_window]
    early.sort(key=lambda t: -(min(t.end, 30) - t.start))
    seeds: list[Tracklet] = []
    for t in early:
        if len(seeds) == n_robots:
            break
        if all(np.hypot(*np.subtract(t.c(0), s.c(0))) > 40 for s in seeds):
            seeds.append(t)
    chains = [Chain(i, "red" if s.c(0)[0] < mid_x else "blue", [s]) for i, s in enumerate(seeds)]
    used = {s.id for s in seeds}
    for t in trs:
        if t.id in used:
            continue
        best, best_cost = None, 1e9
        for ch in chains:
            gap = t.start - ch.end
            if gap < -overlap or gap > max_gap:
                continue
            d = float(np.hypot(*np.subtract(t.c(0), ch.last())))
            if d > base_px + px_per_s * max(gap, 0.0):
                continue
            cost = d + 20.0 * max(gap, 0.0)
            if cost < best_cost:
                best, best_cost = ch, cost
        if best is not None:
            best.parts.append(t)
            used.add(t.id)
        elif len(chains) < n_robots:
            chains.append(Chain(len(chains), "red" if t.c(0)[0] < mid_x else "blue", [t]))
            used.add(t.id)
    return chains


def climb(ch: Chain, t_from: float = 115.0, t_to: float = 152.0, min_rise_px: float = 45.0,
          max_dx_px: float = 45.0) -> dict[str, Any]:
    """Upward motion signature in the end game. Returns rise (px), start/end time."""
    s = [(t, b) for t, b in ch.samples() if t_from <= t <= t_to]
    if len(s) < 10:
        return {"climbed": False, "rise": 0.0, "reason": "not seen in end game"}
    t = np.array([x[0] for x in s])
    top = np.array([x[1][1] for x in s])
    cx = np.array([(x[1][0] + x[1][2]) / 2 for x in s])
    k = 9
    top_s = np.convolve(top, np.ones(k) / k, mode="valid")
    cx_s = np.convolve(cx, np.ones(k) / k, mode="valid")
    ts = t[k // 2: k // 2 + len(top_s)]
    # largest drop in box-top y (= rise in the image) with the minimum after the maximum
    best = (0.0, 0, 0)
    hi = 0
    for i in range(len(top_s)):
        if top_s[i] > top_s[hi]:
            hi = i
        r = top_s[hi] - top_s[i]
        if r > best[0]:
            best = (float(r), hi, i)
    rise, i0, i1 = best
    dx = float(abs(cx_s[i1] - cx_s[i0])) if len(cx_s) else 0.0
    ok = rise >= min_rise_px and dx <= max_dx_px
    return {"climbed": bool(ok), "rise": round(rise, 1), "dx": round(dx, 1),
            "t_start": round(float(ts[i0]), 2) if len(ts) else None,
            "t_end": round(float(ts[i1]), 2) if len(ts) else None,
            "climb_seconds": round(float(ts[i1] - ts[i0]), 2) if ok else None}


def coverage(ch: Chain, step_s: float = 0.1, t0: float = 0.0, t1: float = 150.0) -> float:
    ts = {round(t, 1) for t, _ in ch.samples() if t0 <= t <= t1}
    return len(ts) / ((t1 - t0) / step_s)


def official_end(m: dict[str, Any]) -> dict[str, dict[int, float]]:
    d = m.get("details") or {}
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for side in ("red", "blue"):
        for i, w in enumerate(("One", "Two", "Three"), 1):
            v = d.get(f"{side}Robot{w}Parking")
            if v is not None:
                out[side][i] = float(v)
    return out


def stations(m: dict[str, Any]) -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = defaultdict(dict)
    for p in m["participants"]:
        side = "red" if p["station"] // 10 == 1 else "blue"
        out[side][p["station"] % 10] = p["country"]
    return out


# ---------- v2: linking in field metres with a constant-velocity Kalman filter ----------

def field_track(tr: Tracklet, cam) -> np.ndarray:
    """(n, 3) array of t, x, y in metres from box bottom centres."""
    uv = np.array([[(b[0] + b[2]) / 2, b[3]] for b in tr.box])
    xy = cam.to_field(uv)
    return np.c_[np.array(tr.t), xy]


class CV:
    """Constant-velocity Kalman filter on (x, y, vx, vy)."""

    def __init__(self, x: float, y: float, q: float = 4.0, r: float = 0.15):
        self.s = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([r * r, r * r, 1.0, 1.0])
        self.q, self.r = q, r

    def predict(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        Q = G @ G.T * self.q
        return F @ self.s, F @ self.P @ F.T + Q

    def update(self, dt: float, z: np.ndarray) -> None:
        s, P = self.predict(dt)
        Hm = np.eye(2, 4)
        S = Hm @ P @ Hm.T + np.eye(2) * self.r ** 2
        K = P @ Hm.T @ np.linalg.inv(S)
        self.s = s + K @ (z - Hm @ s)
        self.P = (np.eye(4) - K @ Hm) @ P


def run_filter(ft: np.ndarray, kf: CV | None = None, t_prev: float | None = None) -> tuple[CV, float]:
    for t, x, y in ft:
        if kf is None:
            kf, t_prev = CV(x, y), t
            continue
        kf.update(max(1e-3, t - t_prev), np.array([x, y]))
        t_prev = t
    assert kf is not None and t_prev is not None
    return kf, t_prev


def stitch_metric(trs: list[Tracklet], cam, n_robots: int = 6, seed_window: float = 3.0,
                  max_gap: float = 15.0, gate: float = 4.0, max_speed: float = 2.5,
                  field: tuple[float, float] = (7.0, 7.0)) -> list[Chain]:
    """Same seeding and single-pass order as `stitch`, but in metres with velocity.

    A tracklet joins the chain whose Kalman prediction at the tracklet's first sample is
    closest in Mahalanobis distance, gated at `gate` sigma and by a hard speed limit
    (`max_speed` m/s over the gap, plus 0.4 m). Tracklets whose feet fall more than 0.5 m
    outside the field are ignored (people on the rail, wall signs).
    """
    fts = {t.id: field_track(t, cam) for t in trs}
    ok = [t for t in trs
          if np.median(fts[t.id][:, 1]) > -0.5 and np.median(fts[t.id][:, 1]) < field[0] + 0.5
          and np.median(fts[t.id][:, 2]) > -0.5 and np.median(fts[t.id][:, 2]) < field[1] + 0.5]
    early = sorted([t for t in ok if t.start <= seed_window], key=lambda t: -(min(t.end, 30) - t.start))
    seeds: list[Tracklet] = []
    for t in early:
        if len(seeds) == n_robots:
            break
        if all(np.hypot(*(fts[t.id][0, 1:] - fts[s.id][0, 1:])) > 0.3 for s in seeds):
            seeds.append(t)
    chains = [Chain(i, "red" if fts[s.id][0, 1] < field[0] / 2 else "blue", [s])
              for i, s in enumerate(seeds)]
    filt = {c.slot: run_filter(fts[c.parts[0].id]) for c in chains}
    used = {s.id for s in seeds}
    for t in ok:
        if t.id in used:
            continue
        z = fts[t.id][0, 1:]
        best, best_d = None, 1e9
        for ch in chains:
            kf, tl = filt[ch.slot]
            gap = t.start - tl
            if gap < -0.3 or gap > max_gap:
                continue
            s, P = kf.predict(max(gap, 1e-3))
            if np.hypot(*(z - kf.s[:2])) > 0.4 + max_speed * max(gap, 0.0):
                continue
            S = P[:2, :2] + np.eye(2) * kf.r ** 2
            d = float(np.sqrt((z - s[:2]) @ np.linalg.inv(S) @ (z - s[:2])))
            if d < gate and d < best_d:
                best, best_d = ch, d
        if best is not None:
            best.parts.append(t)
            kf, tl = filt[best.slot]
            filt[best.slot] = run_filter(fts[t.id], kf, tl)
            used.add(t.id)
    return chains
