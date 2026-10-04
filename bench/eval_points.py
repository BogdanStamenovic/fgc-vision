"""Score detections against point labels. A detection box matches a label point if the point
lies inside the box (Hungarian, cost = distance of point to box centre). Boxes wider than
MAXW px are rejected (a box over half the field 'contains' every point).
Reports precision/recall/F1 at the best score threshold per model, plus at a fixed 0.25."""
from __future__ import annotations
import json, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
MAXW = 320

def match(boxes, pts):
    if not boxes or not pts: return 0
    C = np.full((len(boxes), len(pts)), 1e6)
    for i, b in enumerate(boxes):
        cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        for j, (x, y) in enumerate(pts):
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                C[i, j] = np.hypot(x-cx, y-cy)
    r, c = linear_sum_assignment(C)
    return int(sum(C[i, j] < 1e6 for i, j in zip(r, c)))

def inside_any(b, pts):
    return any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for x, y in pts)

def contained(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1e-6, (a[2]-a[0]) * (a[3]-a[1]))

def suppress(bs, frac=0.6):
    """Drop a box when >= frac of it lies inside a higher-scoring box (part-boxes)."""
    keep = []
    for b in sorted(bs, key=lambda d: -d[4]):
        if all(contained(b, k) < frac for k in keep): keep.append(b)
    return keep

POST = True

def score(dets, labels, thr, cls="robot"):
    tp = fp = fn = 0
    for f, lab in labels.items():
        if f.startswith("_"): continue
        bs = [d for d in dets.get(f, []) if d[5] == cls and d[4] >= thr and d[2]-d[0] <= MAXW and d[1] < 860]
        if POST: bs = suppress(bs)
        m = match(bs, lab["robots"])
        extra = [b for b in bs if not inside_any(b, lab["robots"]) and not inside_any(b, lab["unsure"])]
        # boxes containing a labelled point but not matched (duplicates) count as FP
        dup = len(bs) - m - len(extra) - sum(1 for b in bs if not inside_any(b, lab["robots"]) and inside_any(b, lab["unsure"]))
        tp += m; fn += len(lab["robots"]) - m; fp += len(extra) + max(0, dup)
    p = tp/(tp+fp) if tp+fp else 0; r = tp/(tp+fn) if tp+fn else 0
    return p, r, (2*p*r/(p+r) if p+r else 0), tp, fp, fn

def main():
    global POST
    POST = "--raw" not in sys.argv
    res = json.load(open(sys.argv[1])); labels = json.load(open(sys.argv[2]))
    cls = "ball" if "ball" in sys.argv[2] else "robot"
    print(f"{'model':14} {'s/frame':>8} {'VRAM':>6} | best thr  P     R     F1  (tp/fp/fn) | @0.25  P     R")
    for m, d in res.items():
        best = max((score(d["dets"], labels, t, cls) + (t,) for t in np.arange(0.05, 0.9, 0.025)), key=lambda x: x[2])
        p25 = score(d["dets"], labels, 0.25, cls)
        print(f"{m:14} {d['sec_per_frame']:8.3f} {d['vram_peak_mib']:6} | {best[6]:.3f} {best[0]:.2f} {best[1]:.2f} {best[2]:.2f} ({best[3]}/{best[4]}/{best[5]}) | {p25[0]:.2f} {p25[1]:.2f}")

if __name__ == "__main__": main()
