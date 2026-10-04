"""Bigger labelling sheets: header = one clean crop per gallery robot, then 20 fragments
per sheet (4 crops each, 120 px). No predictions are drawn."""
import sys, json, numpy as np, cv2
key = sys.argv[1]; D = "/mnt/offload/fgc-vision/reid"
z = np.load(f"{D}/{key}.npz"); g, fr = z["gal"], z["frag"]; meta = json.loads(str(z["meta"]))
S = 120
hdr = np.full((S + 20, 6 * (S + 8) * 2, 3), 30, np.uint8)
for i in range(6):
    for j, ci in enumerate((0, 11)):
        hdr[20:, i * 2 * (S + 8) + j * S: i * 2 * (S + 8) + (j + 1) * S] = cv2.resize(g[i][ci], (S, S))
    cv2.putText(hdr, f"G{i} {meta['gal'][i]['side']}", (i * 2 * (S + 8), 15), 0, 0.5, (0, 255, 255), 1)
cols = 4
for k in range(0, len(fr), 20):
    n = min(20, len(fr) - k); nr = (n + cols - 1) // cols
    body = np.full((nr * (S + 20), cols * (4 * S + 16), 3), 50, np.uint8)
    for i in range(n):
        r, c = divmod(i, cols); m = meta["frag"][k + i]
        for j, ci in enumerate((0, 2, 5, 7)):
            body[r * (S + 20) + 20:(r + 1) * (S + 20), c * (4 * S + 16) + j * S:c * (4 * S + 16) + (j + 1) * S] = cv2.resize(fr[k + i][ci], (S, S))
        cv2.putText(body, f"F{k+i} {m['t0']:.0f}-{m['t1']:.0f}s @({m['c0'][0]:.0f},{m['c0'][1]:.0f})", (c * (4 * S + 16), r * (S + 20) + 15), 0, 0.5, (255, 255, 255), 1)
    w = max(hdr.shape[1], body.shape[1])
    pad = lambda a: np.pad(a, ((0, 0), (0, w - a.shape[1]), (0, 0)))
    cv2.imwrite(f"{D}/{key}.lab20_{k // 20}.jpg", np.vstack([pad(hdr), pad(body)]), [cv2.IMWRITE_JPEG_QUALITY, 90])
print(len(fr))
