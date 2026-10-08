"""All in-field robot boxes at sample times on one sheet (row = time), plus context frames
for counting visible robots by eye. usage: crop_sheet26.py KEY t1 t2 ..."""
import sys, json, cv2, numpy as np
sys.path.insert(0, "/home/bodas/data/fgc-vision/bench")
import gt26 as G
key = sys.argv[1]; ts = [float(x) for x in sys.argv[2:]]
res, _, _ = G.load(key)
S = 150; rows = []; ctxs = []; meta = {}
for t in ts:
    tt, bs = G.boxes_at(res, t); f = G.frame(key, res, tt)
    tiles = []
    for i, r in enumerate(bs):
        c = G.crop(f, r[1:5], S); cv2.putText(c, f"{tt:.0f}.{i}", (4, 20), 0, 0.6, (0, 255, 255), 2); tiles.append(c)
    lab = np.zeros((S, 70, 3), np.uint8); cv2.putText(lab, f"{tt:.0f}s", (2, 80), 0, 0.8, (255, 255, 255), 2)
    rows.append(np.hstack([lab] + tiles + [np.zeros((S, S, 3), np.uint8)] * (8 - len(tiles))) if len(tiles) <= 8 else np.hstack([lab] + tiles[:8]))
    ctxs.append(G.ctx(f, bs, tt)); meta[f"{tt:.1f}"] = [r[0] for r in bs]
cv2.imwrite(f"/mnt/offload/fgc-vision/frames/gt/{key}_sheet.jpg", np.vstack(rows))
for i in range(0, len(ctxs), 4):
    cv2.imwrite(f"/mnt/offload/fgc-vision/frames/gt/{key}_ctx{i//4}.jpg", np.vstack([np.hstack(ctxs[j:j+2]) for j in range(i, min(i+4, len(ctxs)), 2) if len(ctxs[j:j+2]) == 2] or [ctxs[i]]))
json.dump(meta, open(f"/mnt/offload/fgc-vision/frames/gt/{key}_sheet.json", "w"))
print(meta)
