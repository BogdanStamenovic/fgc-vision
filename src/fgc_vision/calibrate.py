"""Semi-automatic camera calibration for the fixed field cameras.

Inputs per camera: one clip and four ROUGH corner hints (image px, near-left, far-left,
far-right, near-right; ±30 px is enough, corners hidden behind towers are fine).

1. Median of ~60 frames over the clip: robots and most people vanish.
2. Green-carpet mask (HSV) -> largest component -> its boundary pixels.
3. Each boundary pixel is assigned to the nearest of the 4 hint edges; pixels farther than
   12 px from that edge (tower bases, people, robots) are dropped.
4. Unknowns: radial distortion k1 (division model about the image centre) and the
   homography H. Residual per boundary pixel: its mapped field coordinate minus the rail's
   (near y=0, back y=W, left x=0, right x=W), in metres. Initialised from the 4 hint corners
   with k1=0, solved with scipy least_squares (soft-L1).
5. Check, not used in the fit: red and blue tape pixels (REGIONAL ZONE, 50 cm from the side
   rails, 410 cm long) mapped to the field. Reported as median x and the y-extent.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

W = 7.0


def median_frame(clip: str, n: int = 60) -> np.ndarray:
    cap = cv2.VideoCapture(clip)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fs = []
    for fi in np.linspace(0, tot - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, f = cap.read()
        if ok:
            fs.append(f)
    return np.median(np.stack(fs), axis=0).astype(np.uint8)


def undistort_pts(uv: np.ndarray, k1: float, c: tuple[float, float] = (960, 540),
                  s: float = 1000.0) -> np.ndarray:
    d = (uv - np.array(c)) / s
    r2 = (d ** 2).sum(1, keepdims=True)
    return d / (1 + k1 * r2) * s + np.array(c)


def apply(params: np.ndarray, uv: np.ndarray) -> np.ndarray:
    k1 = params[0]
    H = np.append(params[1:], 1.0).reshape(3, 3)
    u = undistort_pts(uv, k1)
    p = np.c_[u, np.ones(len(u))] @ H.T
    return p[:, :2] / p[:, 2:3]


def carpet_edges(img: np.ndarray, hints: np.ndarray, tol: float = 12.0, fixed: bool = False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # carpet colour differs per field and lighting (green, teal, dark green): learn it from
    # the inner 60 % of the hint quadrilateral
    c = hints.mean(0)
    inner = (c + 0.6 * (hints - c)).astype(np.int32)
    m0 = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(m0, [inner], 255)
    h0, s0, v0 = np.median(hsv[m0 > 0], axis=0)
    if fixed:   # the d1f1-validated range for the green 2025 carpet
        green = cv2.inRange(hsv, (45, 60, 50), (100, 255, 255))
    else:
        green = cv2.inRange(hsv, (max(0, h0 - 10), max(25, s0 * 0.55), max(20, v0 * 0.45)),
                            (min(180, h0 + 10), 255, 255))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    _n, lab, stats, _ = cv2.connectedComponentsWithStats(green)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (lab == big).astype(np.uint8) * 255
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(cs, key=len).reshape(-1, 2).astype(float)
    # edges: near (NL-NR), left (NL-FL), back (FL-FR), right (FR-NR)
    nl, fl, fr, nr = hints
    edges = {"near": (nl, nr), "left": (nl, fl), "back": (fl, fr), "right": (fr, nr)}
    out = {}
    for name, (a, b) in edges.items():
        ab = b - a
        t = ((pts - a) @ ab) / (ab @ ab)
        proj = a + np.outer(t, ab)
        d = np.linalg.norm(pts - proj, axis=1)
        sel = (t > 0.05) & (t < 0.95) & (d < tol)
        out[name] = pts[sel]
    return out, mask


def carpet_contour(mask: np.ndarray) -> np.ndarray:
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cs, key=len).reshape(-1, 2).astype(float)


def tape_pixels(img: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    inner = cv2.erode(mask, np.ones((15, 15), np.uint8))
    red = (np.asarray(cv2.inRange(hsv, (0, 120, 120), (8, 255, 255))) |
           np.asarray(cv2.inRange(hsv, (170, 120, 120), (180, 255, 255))))
    blue = np.asarray(cv2.inRange(hsv, (100, 140, 120), (125, 255, 255)))
    out = {}
    for k, m in (("red", red), ("blue", blue)):
        m = np.asarray(m) & np.asarray(inner)
        ys, xs = np.nonzero(m)
        out[k] = np.c_[xs, ys].astype(float)
    return out


def calibrate(clip: str, hints: list[list[float]], out: Path | None = None,
              mode: str = "iter") -> dict:
    """mode 'v0': fixed green range, 12 px hint tolerance, weak corner prior, single solve
    (validated on d1f1). mode 'iter': adaptive carpet colour, bounded k1, stronger prior,
    two re-assignment passes. Neither works on every camera; check `tape_check`."""
    img = median_frame(clip)
    H = np.asarray(hints, float)
    v0 = mode == "v0"
    edges, mask = carpet_edges(img, H, tol=12.0 if v0 else 25.0, fixed=v0)
    edges_all = {"all": carpet_contour(mask)}
    H0, _ = cv2.findHomography(H, np.array([[0, 0], [0, W], [W, W], [W, 0]], float))
    H0 = H0 / H0[2, 2]
    p0 = np.r_[0.0, H0.flatten()[:8]]
    target = {"near": (1, 0.0), "back": (1, W), "left": (0, 0.0), "right": (0, W)}

    def resid(p):
        r = []
        for name, (axis, val) in target.items():
            if len(edges[name]):
                r.append(apply(p, edges[name])[:, axis] - val)
        # weak prior keeping the 4 hint corners near the field corners (fixes the
        # along-rail sliding freedom when a rail is mostly occluded)
        r.append((0.05 if v0 else 0.3) * (apply(p, H) - np.array([[0, 0], [0, W], [W, W], [W, 0]])).ravel())
        return np.concatenate(r)

    lb = np.r_[-0.3, np.full(8, -np.inf)]
    ub = np.r_[0.3, np.full(8, np.inf)]
    if v0:
        lb, ub = np.full(9, -np.inf), np.full(9, np.inf)
    sol = least_squares(resid, p0, loss="soft_l1", f_scale=0.05, bounds=(lb, ub))
    # re-assign boundary pixels to rails with the current model, then refit (twice)
    allpts: np.ndarray = np.concatenate(list(edges_all.values()))
    for _ in range(0 if v0 else 2):
        xy = apply(sol.x, allpts)
        inb = (xy[:, 0] > -0.3) & (xy[:, 0] < W + 0.3) & (xy[:, 1] > -0.3) & (xy[:, 1] < W + 0.3)
        edges = {"near": allpts[inb & (np.abs(xy[:, 1]) < 0.25) & (xy[:, 0] > 0.3) & (xy[:, 0] < W - 0.3)],
                 "back": allpts[inb & (np.abs(xy[:, 1] - W) < 0.25) & (xy[:, 0] > 0.3) & (xy[:, 0] < W - 0.3)],
                 "left": allpts[inb & (np.abs(xy[:, 0]) < 0.25) & (xy[:, 1] > 0.3) & (xy[:, 1] < W - 0.3)],
                 "right": allpts[inb & (np.abs(xy[:, 0] - W) < 0.25) & (xy[:, 1] > 0.3) & (xy[:, 1] < W - 0.3)]}
        edges = {k: (v if len(v) >= 30 else v[:0]) for k, v in edges.items()}
        sol = least_squares(resid, sol.x, loss="soft_l1", f_scale=0.05, bounds=(lb, ub))
    res = {}
    for name, (axis, val) in target.items():
        if len(edges[name]):
            e = apply(sol.x, edges[name])[:, axis] - val
            res[name] = {"n": len(e), "median_abs_m": round(float(np.median(np.abs(e))), 3)}
    sol0 = least_squares(lambda p: resid(np.r_[0.0, p]), sol.x[1:], loss="soft_l1", f_scale=0.05)
    res0 = {}
    for name, (axis, val) in target.items():
        if len(edges[name]):
            e = apply(np.r_[0.0, sol0.x], edges[name])[:, axis] - val
            res0[name] = round(float(np.median(np.abs(e))), 3)
    check = {}
    for k, px in tape_pixels(img, mask).items():
        for tag, p in (("k1", sol.x), ("nok1", np.r_[0.0, sol0.x])):
            if len(px) < 50:
                continue
            xy = apply(p, px)
            side = xy[:, 0] < W / 2 if k == "red" else xy[:, 0] > W / 2
            xy = xy[side & (xy[:, 1] > -0.2) & (xy[:, 1] < W + 0.2)]
            if len(xy) < 50:
                continue
            # long tape line = the pixels farthest from the rail (inner edge of the zone)
            dist = xy[:, 0] if k == "red" else W - xy[:, 0]
            longline = xy[(dist > 0.3) & (dist < 0.8)]
            check[f"{k}_{tag}"] = {
                "line_dist_from_rail_m": round(float(np.median(
                    longline[:, 0] if k == "red" else W - longline[:, 0])), 3)
                if len(longline) else None,
                "line_y_extent_m": round(float(np.percentile(longline[:, 1], 98) -
                                               np.percentile(longline[:, 1], 2)), 2)
                if len(longline) else None}
    d = {"k1": float(sol.x[0]), "H": [float(v) for v in np.append(sol.x[1:], 1.0)],
         "hints": hints, "rail_residuals": res, "rail_residuals_no_k1_m": res0,
         "tape_check": check, "expected": {"line_dist_from_rail_m": 0.5,
                                           "line_y_extent_m": 4.1}}
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(d, indent=1))
        dbg = img.copy()
        dbg[mask == 0] = (dbg[mask == 0] * 0.4).astype(np.uint8)
        for name, pts in edges.items():
            for x, y in pts[::3].astype(int):
                cv2.circle(dbg, (x, y), 1, (0, 0, 255), -1)
        # field grid every metre, projected back
        grid = []
        for v in np.arange(0, W + 0.01, 1.0):
            grid.append(np.c_[np.full(50, v), np.linspace(0, W, 50)])
            grid.append(np.c_[np.linspace(0, W, 50), np.full(50, v)])
        for g in grid:
            uv = invert(sol.x, g)
            cv2.polylines(dbg, [uv.astype(np.int32)], False, (0, 255, 255), 1)
        cv2.imwrite(str(out.with_suffix(".jpg")), dbg)
    return d


def invert(params: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Field -> image (numerically; used for drawing)."""
    H = np.append(params[1:], 1.0).reshape(3, 3)
    p = np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(H).T
    u = p[:, :2] / p[:, 2:3]
    k1, c, s = params[0], np.array([960, 540]), 1000.0
    d = (u - c) / s
    x = d.copy()
    for _ in range(20):   # solve d = x / (1 + k1 |x|^2)
        x = d * (1 + k1 * (x ** 2).sum(1, keepdims=True))
    return x * s + c
