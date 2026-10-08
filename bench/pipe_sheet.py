"""Contact sheet of one pipe's region, every STEP s from T0 to the end, for hand-timing climbs.
usage: pipe_sheet.py KEY red|blue T0 STEP"""
import sys, json, cv2, numpy as np
sys.path.insert(0, "/home/bodas/data/fgc-vision/bench"); import gt26 as G
key, side, t0, step = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
P = json.load(open("/mnt/offload/fgc-vision/2026/pipes.json"))[key][side]
x0, x1 = min(P[0], P[2]) - 140, max(P[0], P[2]) + 140; y0, y1 = min(P[1], P[3]) - 90, max(P[1], P[3]) + 60
res, _, _ = G.load(key); cap = cv2.VideoCapture(str(G.D / f"{key}.mp4")); tiles = []
t = t0
while t <= 152:
    cap.set(1, int(res["start_frame"] + t * res["fps"])); f = cap.read()[1]
    c = f[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)].copy()
    c = cv2.resize(c, (320, int(320 * c.shape[0] / c.shape[1])))
    cv2.putText(c, f"{t:.0f}", (4, 20), 0, 0.7, (0, 255, 255), 2); tiles.append(c); t += step
while len(tiles) % 6: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(f"/mnt/offload/fgc-vision/frames/gt/{key}_{side}pipe.jpg", np.vstack([np.hstack(tiles[i:i+6]) for i in range(0, len(tiles), 6)]))
