"""Contact sheet of a region of a clip, match time T0..T1 every STEP s.
usage: srb_sheet.py CLIP START_FRAME OUT x0,y0,x1,y1 T0 T1 STEP [width]"""
import sys, cv2, numpy as np
clip, s0, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
x0, y0, x1, y1 = map(int, sys.argv[4].split(",")); t0, t1, st = map(float, sys.argv[5:8])
W = int(sys.argv[8]) if len(sys.argv) > 8 else 320
cap = cv2.VideoCapture(clip); tiles = []; t = t0
while t <= t1 + 1e-6:
    cap.set(1, int(s0 + t * 60)); ok, f = cap.read()
    if not ok: break
    c = f[y0:y1, x0:x1].copy(); c = cv2.resize(c, (W, int(W * (y1 - y0) / (x1 - x0))))
    cv2.putText(c, f"{t:.0f}", (4, 22), 0, 0.7, (0, 255, 255), 2); tiles.append(c); t += st
cols = max(1, 1920 // W)
while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))
