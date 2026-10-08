"""One row per track >= MIN s: 10 crops over its life (in-field samples, t in [0,150])."""
import sys, json, cv2, numpy as np
sys.path.insert(0, "/home/bodas/data/fgc-vision/bench"); import gt26 as G
key = sys.argv[1]; MIN = float(sys.argv[2]) if len(sys.argv) > 2 else 20
res, _, _ = G.load(key); T = {}
for fr in res["frames"]:
    if not (0 <= fr["t"] <= 152): continue
    for r in fr["robots"]:
        if cv2.pointPolygonTest(G.POLY, ((r[1]+r[3])/2, (r[2]+r[4])/2), False) >= 0: T.setdefault(r[0], []).append((fr["t"], r[1:5]))
cap = cv2.VideoCapture(str(G.D / f"{key}.mp4")); S = 120; rows = []; ids = []
for tid, v in sorted(T.items(), key=lambda x: -len(x[1])):
    if len(v) / 10 < MIN: continue
    row = []
    for i in np.linspace(0, len(v) - 1, 10).astype(int):
        t, b = v[i]; cap.set(1, int(round(res["start_frame"] + t * res["fps"]))); f = cap.read()[1]
        c = G.crop(f, b, S); cv2.putText(c, f"{t:.0f}", (3, 15), 0, 0.5, (0, 255, 255), 2); row.append(c)
    lab = np.zeros((S, 90, 3), np.uint8); cv2.putText(lab, f"#{len(ids)}", (2, 40), 0, 0.8, (255, 255, 255), 2); cv2.putText(lab, f"{len(v)/10:.0f}s", (2, 80), 0, 0.6, (200, 200, 200), 1)
    rows.append(np.hstack([lab] + row)); ids.append(tid)
cv2.imwrite(f"/mnt/offload/fgc-vision/frames/gt/{key}_long.jpg", np.vstack(rows))
json.dump(ids, open(f"/mnt/offload/fgc-vision/frames/gt/{key}_long.json", "w")); print(ids)
