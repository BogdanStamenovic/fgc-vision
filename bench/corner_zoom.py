"""Zoom full-res frame around given points (2x, 10 px grid) for precise calibration clicks.
usage: corner_zoom.py FRAME.png OUT.jpg x,y [x,y ...]"""
import sys, cv2, numpy as np
f = cv2.imread(sys.argv[1]); tiles = []
for s in sys.argv[3:]:
    x, y = map(int, s.split(",")); r = 60
    x0, y0 = max(0, x - r), max(0, y - r)
    c = cv2.resize(f[y0:y0 + 2 * r, x0:x0 + 2 * r], (480, 480), interpolation=cv2.INTER_CUBIC)
    for k in range(0, 2 * r + 1, 10):
        col = (0, 255, 255) if (x0 + k) % 50 == 0 else (0, 120, 140)
        cv2.line(c, (k * 4, 0), (k * 4, 480), col, 1)
        if (x0 + k) % 50 == 0: cv2.putText(c, str(x0 + k), (k * 4 + 2, 12), 0, 0.4, (0, 255, 255), 1)
        col = (0, 255, 255) if (y0 + k) % 50 == 0 else (0, 120, 140)
        cv2.line(c, (0, k * 4), (480, k * 4), col, 1)
        if (y0 + k) % 50 == 0: cv2.putText(c, str(y0 + k), (2, k * 4 - 2), 0, 0.4, (0, 255, 255), 1)
    tiles.append(c)
cv2.imwrite(sys.argv[2], np.hstack(tiles))
