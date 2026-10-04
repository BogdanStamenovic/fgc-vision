"""Pre-start frame (t=-5 s) with tracked robot boxes and ids, for confirming start spots."""
import json, sys, cv2
key = sys.argv[1]; D = "/mnt/offload/fgc-vision"
res = json.load(open(f"{D}/tracks/{key}.pre.json"))
fr = min(res["frames"], key=lambda f: abs(f["t"] + 5))
cap = cv2.VideoCapture(f"{D}/video/2025_{key}.mp4"); cap.set(cv2.CAP_PROP_POS_FRAMES, fr["f"]); ok, im = cap.read()
for tid, x0, y0, x1, y1, c in fr["robots"]:
    cv2.rectangle(im, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 255), 2)
    cv2.putText(im, f"{tid}:{c:.2f}", (int(x0), int(y0) - 4), 0, 0.6, (255, 255, 255), 2)
for x in range(0, 1920, 100):
    cv2.putText(im, str(x), (x, 270), 0, 0.5, (0, 255, 255), 1)
cv2.imwrite(f"{D}/frames/look/start_{key}.jpg", im[250:860, :])
print(fr["t"], fr["robots"])
