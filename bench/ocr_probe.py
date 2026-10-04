"""Probe: tesseract on overlay timer/score ROIs, 1 frame per second of a clip."""
import subprocess, sys, cv2, numpy as np
ROI = {"timer": (850, 855, 1070, 950), "red": (855, 960, 955, 1055), "blue": (965, 960, 1065, 1055)}
def ocr(img, wl):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if b.mean() < 128: b = 255 - b
    b = cv2.copyMakeBorder(b, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    ok, png = cv2.imencode(".png", b)
    p = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "7", "-c", f"tessedit_char_whitelist={wl}"],
                       input=png.tobytes(), capture_output=True)
    return p.stdout.decode().strip()
cap = cv2.VideoCapture(sys.argv[1]); fps = cap.get(cv2.CAP_PROP_FPS); i = 0
while True:
    ok, f = cap.read()
    if not ok: break
    if i % int(fps) == 0:
        r = {k: ocr(f[y0:y1, x0:x1], "0123456789:" if k == "timer" else "0123456789") for k, (x0, y0, x1, y1) in ROI.items()}
        print(f"{i/fps:6.1f}", r["timer"], r["red"], r["blue"])
    i += 1
