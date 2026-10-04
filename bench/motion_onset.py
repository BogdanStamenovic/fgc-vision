"""Motion energy inside the field polygon (floor only, below the rails' top) over the first
40 s of each clip, at 10 Hz; prints the series coarsely and the first sustained jump."""
import sys, json, cv2, numpy as np
sys.path.insert(0, "/home/bodas/data/fgc-vision/src")
from fgc_vision import identity as I
D = "/mnt/offload/fgc-vision"
mask = np.zeros((270, 480), np.uint8)
cv2.fillPoly(mask, [(I.FIELD_POLY_2025 / 4).astype(np.int32)], 1)
for key in sys.argv[1:]:
    cap = cv2.VideoCapture(f"{D}/video/2025_{key}.mp4"); fps = cap.get(5)
    prev = None; e = []; i = 0
    while i < 40 * fps:
        ok, f = cap.read()
        if not ok: break
        if i % 6 == 0:
            g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(f, (480, 270)), cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
            if prev is not None:
                d = (np.abs(g - prev) > 18) & (mask > 0); e.append(d.sum())
            prev = g
        i += 1
    e = np.array(e, float); t = (np.arange(len(e)) + 1) * 6 / fps
    base = np.median(e[:40]); thr = base * 3 + 150
    on = next((t[k] for k in range(len(e) - 10) if (e[k:k + 10] > thr).mean() > 0.8), None)
    sy = json.load(open(f"{D}/tracks/{key}.sync.json")) if __import__("os").path.exists(f"{D}/tracks/{key}.sync.json") else None
    meta = json.load(open(f"{D}/video/2025_{key}.json"))
    print(key, "motion onset clip-s", on, "| overlay t0", sy and round(sy["start_frame"] / sy["fps"], 2),
          "| matchwatch", round(meta["matchwatch_start"] - meta["stream_offset"], 2),
          "| series(1s):", [int(x) for x in e[::10][:30]])
